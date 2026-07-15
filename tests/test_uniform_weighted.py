# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Tests for the UNIFORM weighted layer (UT3Weights + absorb / weighted_norm / weighted_inner).

The ragged weighted layer is the oracle: per docs/uniform_equivalence_contract.md, a uniform op is
correct iff ``to_ragged(op_uniform(to_uniform(x), ...)) == op_ragged(x, ...)`` on the REAL (masked)
parts -- garbage padding is don't-care. Per docs/contributor/testing_strategy.md that is necessary but
NOT sufficient (dense/ragged agreement is blind to a too-permissive mask), so each op also gets:

  1. equivalence to ragged, over structures x stacks incl. VARYING RANKS across the stack;
  2. garbage-padded-input robustness (large-FINITE garbage: the layer masks by multiply, so nan would
     poison rather than test);
  3. exact output-mask assertions, derived independently of the implementation;
  4. jax/jit dispatch (masks stay host numpy).

Numerical correctness is checked numpy-only (the backend is backend-agnostic); jax is covered by the
dispatch test at the bottom and by tests/test_dispatch.py.
"""
import unittest
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ut3_conversions as ut3_conversions
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.common as common

# (shape, tucker_ranks, tt_ranks)
STRUCTURES = [
    ((6, 7, 8), (2, 3, 2), (1, 2, 2, 1)),
    ((5, 6, 7, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)),
]

# (n, r) padding overrides: None = tight (padding only where ranks differ), else FORCED-LARGER so that
# every edge carries a padded region -- the case that actually exercises the mask.
PADS = [None, (5, 4)]


def rand_ragged_weights(x, rng):
    """A random ragged T3Weights fitting x. Strictly positive, so reciprocal/sqrt are well-defined on the
    REAL slots and any inf/nan can only have come from the padding."""
    return t3.T3Weights(
        tuple(np.abs(rng.standard_normal(x.stack_shape + (n,))) + 0.5 for n in x.tucker_ranks),
        tuple(np.abs(rng.standard_normal(x.stack_shape + (r,))) + 0.5 for r in x.tt_ranks))


def to_uniform_pair(x, W, pad):
    """(x, W) -> (ux, uW) padded consistently. The weight must share the train's (n, r)."""
    if pad is None:
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    else:
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=pad[0], r=pad[1])
    return ux, ut3.UT3Weights.from_t3weights(W, n=ux.n, r=ux.r)


def corrupt(weights, value=1e6):
    """Write large-FINITE garbage into every padded slot of a UT3Weights (the phantom-rank tripwire).
    Large-finite, not nan: the layer masks by multiplication, so nan would poison rather than test."""
    tkm, ttm = weights.masks.data
    return ut3.UT3Weights(np.where(tkm, weights.tucker_weight_supercore, value),
                          np.where(ttm, weights.tt_weight_supercore, value),
                          weights.masks)


class TestUT3WeightsConversion(unittest.TestCase):
    def test_roundtrip_and_masks(self):
        """to_uniform -> to_ragged recovers the ragged weight exactly, and the masks are the independently
        derived prefix masks (NOT read back from the implementation)."""
        rng = np.random.default_rng(0)
        for struct in STRUCTURES:
            for pad in PADS:
                with self.subTest(struct=struct, pad=pad):
                    x = t3.TuckerTensorTrain.randn(*struct)
                    W = rand_ragged_weights(x, rng)
                    ux, uW = to_uniform_pair(x, W, pad)

                    back = uW.to_t3weights()
                    for fam_got, fam_ref in zip(back.data, W.data):
                        for a, b in zip(fam_got, fam_ref):
                            self.assertTrue(np.array_equal(a, b))

                    # Exact masks, derived from the STRUCTURE's ranks, not from uW.
                    n, r = ux.n, ux.r
                    want_tk = np.array([[j < k for j in range(n)] for k in struct[1]])
                    want_tt = np.array([[j < k for j in range(r)] for k in struct[2]])
                    self.assertTrue(np.array_equal(uW.masks.tucker_edge_mask, want_tk))
                    self.assertTrue(np.array_equal(uW.masks.tt_edge_mask, want_tt))
                    self.assertTrue(np.issubdtype(uW.masks.tucker_edge_mask.dtype, np.bool_))

    def test_masks_are_host_numpy(self):
        """Masks are static structure and stay HOST numpy even when the supercores are jax."""
        if not common.jax_available:
            self.skipTest('jax not available')
        import jax.numpy as jnp
        rng = np.random.default_rng(1)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        W = rand_ragged_weights(x, rng)
        _, uW = to_uniform_pair(x, W, None)
        jW = ut3.UT3Weights(jnp.asarray(uW.tucker_weight_supercore),
                            jnp.asarray(uW.tt_weight_supercore), uW.masks)
        self.assertTrue(common.is_numpy_ndarray(jW.masks.tucker_edge_mask))
        self.assertTrue(common.is_numpy_ndarray(jW.reciprocal().masks.tt_edge_mask))

    def test_varying_ranks_across_stack(self):
        """A varying-rank stack (the determinantal variety) round-trips to a TREE of ragged weights, one
        per element, each with its own ranks."""
        rng = np.random.default_rng(2)
        xs = [t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1)),
              t3.TuckerTensorTrain.randn((6, 7, 8), (1, 2, 3), (1, 1, 2, 1))]
        Ws = [rand_ragged_weights(xi, rng) for xi in xs]
        uWs = [ut3.UT3Weights.from_t3weights(W, n=3, r=2) for W in Ws]
        stacked = ut3.UT3Weights(np.stack([w.tucker_weight_supercore for w in uWs], axis=1),
                                 np.stack([w.tt_weight_supercore for w in uWs], axis=1),
                                 ut3.UT3Masks(
                                     np.stack([w.masks.tucker_edge_mask for w in uWs], axis=1),
                                     np.stack([w.masks.tt_edge_mask for w in uWs], axis=1)))
        self.assertEqual(stacked.stack_shape, (2,))
        tree = stacked.to_t3weights()
        self.assertEqual(len(tree), 2)
        for got, ref in zip(tree, Ws):
            self.assertEqual(got.tucker_ranks, ref.tucker_ranks)
            for fam_got, fam_ref in zip(got.data, ref.data):
                for a, b in zip(fam_got, fam_ref):
                    self.assertTrue(np.allclose(a, b))


class TestUT3AbsorbAndNorm(unittest.TestCase):
    def test_absorb_equals_ragged(self):
        """to_ragged(absorb_u(to_uniform(x), to_uniform(W))) == absorb_ragged(x, W), on the real parts."""
        rng = np.random.default_rng(3)
        for struct in STRUCTURES:
            for pad in PADS:
                with self.subTest(struct=struct, pad=pad):
                    x = t3.TuckerTensorTrain.randn(*struct)
                    W = rand_ragged_weights(x, rng)
                    ux, uW = to_uniform_pair(x, W, pad)

                    got = ut3.absorb_weights(ux, uW).to_t3()
                    ref = t3.absorb_weights(x, W)
                    for fam_got, fam_ref in zip(got.data, ref.data):
                        for a, b in zip(fam_got, fam_ref):
                            self.assertTrue(np.allclose(a, b, atol=1e-12))

    def test_absorb_preserves_masks_and_shape(self):
        """absorb is shape- and rank-preserving: the output masks are EXACTLY the input's."""
        rng = np.random.default_rng(4)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux, uW = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        xw = ut3.absorb_weights(ux, uW)
        self.assertEqual(xw.masks, ux.masks)
        self.assertEqual(xw.shape, ux.shape)
        self.assertEqual(xw.tucker_supercore.shape, ux.tucker_supercore.shape)

    def test_weighted_norm_inner_equal_ragged(self):
        """weighted_norm / weighted_inner match the ragged twins."""
        rng = np.random.default_rng(5)
        for struct in STRUCTURES:
            for pad in PADS:
                with self.subTest(struct=struct, pad=pad):
                    xA = t3.TuckerTensorTrain.randn(*struct)
                    xB = t3.TuckerTensorTrain.randn(*struct)
                    WA, WB = rand_ragged_weights(xA, rng), rand_ragged_weights(xB, rng)
                    uxA, uWA = to_uniform_pair(xA, WA, pad)
                    uxB, uWB = to_uniform_pair(xB, WB, pad)

                    ref_n = t3.weighted_norm(xA, WA)
                    got_n = ut3.weighted_norm(uxA, uWA)
                    self.assertLess(abs(got_n - ref_n), 1e-9 * (abs(ref_n) + 1))

                    ref_i = t3.weighted_inner(xA, WA, xB, WB)
                    got_i = ut3.weighted_inner(uxA, uWA, uxB, uWB)
                    self.assertLess(abs(got_i - ref_i), 1e-9 * (abs(ref_i) + 1))

    def test_garbage_padding_is_inert(self):
        """The contract says padding is don't-care, so large-finite garbage in the WEIGHT's padding must
        not move any real result: absorb (on the real parts), norm, or inner."""
        rng = np.random.default_rng(6)
        for struct in STRUCTURES:
            with self.subTest(struct=struct):
                x = t3.TuckerTensorTrain.randn(*struct)
                W = rand_ragged_weights(x, rng)
                ux, uW = to_uniform_pair(x, W, (5, 4))
                dirty = corrupt(uW)

                clean_cores = ut3.absorb_weights(ux, uW).to_t3().data
                dirty_cores = ut3.absorb_weights(ux, dirty).to_t3().data
                for fam_c, fam_d in zip(clean_cores, dirty_cores):
                    for a, b in zip(fam_c, fam_d):
                        self.assertTrue(np.array_equal(a, b))   # real parts: bit-identical

                self.assertLess(abs(ut3.weighted_norm(ux, dirty) - ut3.weighted_norm(ux, uW)),
                                1e-9 * (ut3.weighted_norm(ux, uW) + 1))
                self.assertLess(abs(ut3.weighted_inner(ux, dirty, ux, dirty)
                                    - ut3.weighted_inner(ux, uW, ux, uW)),
                                1e-8 * (abs(ut3.weighted_inner(ux, uW, ux, uW)) + 1))


class TestUT3WeightsElementwise(unittest.TestCase):
    def test_reciprocal_sqrt_match_ragged_and_stay_finite(self):
        """reciprocal/sqrt match ragged on the real slots AND keep the padding finite. The padding guard is
        the point: a canonical weight's padding is 0, and a naive 1/w would make it inf -- which then
        poisons every masked reduction (0*inf = nan)."""
        rng = np.random.default_rng(7)
        for struct in STRUCTURES:
            with self.subTest(struct=struct):
                x = t3.TuckerTensorTrain.randn(*struct)
                W = rand_ragged_weights(x, rng)
                _, uW = to_uniform_pair(x, W, (5, 4))

                for name in ('reciprocal', 'sqrt'):
                    got = getattr(uW, name)()
                    ref = getattr(W, name)()
                    for fam_got, fam_ref in zip(got.to_t3weights().data, ref.data):
                        for a, b in zip(fam_got, fam_ref):
                            self.assertTrue(np.allclose(a, b))
                    self.assertTrue(np.isfinite(got.tucker_weight_supercore).all())
                    self.assertTrue(np.isfinite(got.tt_weight_supercore).all())
                    self.assertEqual(got.masks, uW.masks)          # elementwise ops preserve the masks
                    tkm, ttm = uW.masks.data                       # ...and leave canonical zero padding
                    self.assertTrue((got.tucker_weight_supercore[~tkm] == 0).all())
                    self.assertTrue((got.tt_weight_supercore[~ttm] == 0).all())

    def test_naive_reciprocal_would_blow_up(self):
        """The hazard the guard exists for is real, not theoretical: dividing the padded supercore
        directly DOES produce inf, and absorbing that gives nan. (If this ever stops being true the guard
        may look unnecessary -- it is not.)"""
        rng = np.random.default_rng(8)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux, uW = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))

        with np.errstate(divide='ignore'):
            naive = ut3.UT3Weights(1.0 / uW.tucker_weight_supercore,
                                   1.0 / uW.tt_weight_supercore, uW.masks)
        self.assertTrue(np.isinf(naive.tucker_weight_supercore).any())
        self.assertTrue(np.isnan(ut3.absorb_weights(ux, naive).to_dense()).any())   # 0 * inf = nan

        guarded = uW.reciprocal()
        self.assertTrue(np.isfinite(guarded.tucker_weight_supercore).all())
        self.assertTrue(np.isfinite(ut3.absorb_weights(ux, guarded).to_dense()).all())

    def test_reciprocal_does_not_guard_real_zeros(self):
        """Deliberate asymmetry: a genuinely zero weight in a REAL slot still gives inf, exactly as in the
        ragged layer. It is a real weight, not a padding artifact -- silently clamping it would hide a
        rank-deficient point."""
        rng = np.random.default_rng(9)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        _, uW = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        zeroed = ut3.UT3Weights(np.zeros_like(uW.tucker_weight_supercore),
                                uW.tt_weight_supercore, uW.masks)
        with np.errstate(divide='ignore'):
            out = zeroed.reciprocal()
        self.assertTrue(np.isinf(out.tucker_weight_supercore[uW.masks.tucker_edge_mask]).all())


class TestUT3WeightsPreconditions(unittest.TestCase):
    def test_mask_mismatch_is_rejected(self):
        """The precondition uniform ADDS: a weight declaring different edge masks has the same padded
        shape, so nothing errors naturally -- absorbing it would silently zero a real slot. Every
        (train, weights) op must reject it.

        Perturb EACH mask family in turn: a predicate that checked only one of the two would otherwise
        slip through (it did -- caught by mutation testing)."""
        rng = np.random.default_rng(10)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux, uW = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        self.assertTrue(uW.is_consistent_with(ux))

        def perturbed(family):
            """uW with one extra real slot claimed on the given mask family (shape unchanged)."""
            tkm, ttm = (m.copy() for m in uW.masks.data)
            {'tucker': tkm, 'tt': ttm}[family][0, -1] = True
            return ut3.UT3Weights(uW.tucker_weight_supercore, uW.tt_weight_supercore, ut3.UT3Masks(tkm, ttm))

        for family in ('tucker', 'tt'):
            bad = perturbed(family)
            with self.subTest(family=family):
                # The padded shapes are identical -- only the declared masks differ. Nothing would error.
                self.assertEqual(bad.tucker_weight_supercore.shape, uW.tucker_weight_supercore.shape)
                self.assertEqual(bad.tt_weight_supercore.shape, uW.tt_weight_supercore.shape)
                self.assertFalse(bad.is_consistent_with(ux))
                for name, op in (('absorb_weights', lambda: ut3.absorb_weights(ux, bad)),
                                 ('weighted_norm', lambda: ut3.weighted_norm(ux, bad)),
                                 ('weighted_inner', lambda: ut3.weighted_inner(ux, bad, ux, uW))):
                    with self.subTest(op=name), self.assertRaises(ValueError):
                        op()

    def test_wrong_pad_is_rejected(self):
        """A weight padded to a different (n, r) than the train does not fit."""
        rng = np.random.default_rng(11)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux, _ = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        mismatched = ut3.UT3Weights.from_t3weights(rand_ragged_weights(x, rng), n=6, r=4)
        self.assertFalse(mismatched.is_consistent_with(ux))
        with self.assertRaises(ValueError):
            ut3.absorb_weights(ux, mismatched)

    def test_validate_raises(self):
        """Structural inconsistency raises at construction (mask/supercore shape disagreement)."""
        rng = np.random.default_rng(12)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        _, uW = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        with self.assertRaises(ValueError):
            ut3.UT3Weights(uW.tucker_weight_supercore[..., :-1], uW.tt_weight_supercore, uW.masks)


class TestUT3WeightsCombine(unittest.TestCase):
    """concatenate (the + combine) and kronecker (the Hadamard combine) -- the two rank-changing ops, and
    the only ones whose output masks go GAPPY. Per docs/contributor/testing_strategy.md the exact mask
    pattern is asserted, not just the rank count: a too-permissive mask is invisible to value tests."""

    def test_concatenate_equals_ragged(self):
        rng = np.random.default_rng(20)
        for struct in STRUCTURES:
            for pad in PADS:
                with self.subTest(struct=struct, pad=pad):
                    x = t3.TuckerTensorTrain.randn(*struct)
                    WA, WB = rand_ragged_weights(x, rng), rand_ragged_weights(x, rng)
                    _, uA = to_uniform_pair(x, WA, pad)
                    _, uB = to_uniform_pair(x, WB, pad)

                    got = uA.concatenate(uB).to_t3weights()
                    ref = WA.concatenate(WB)
                    for fam_got, fam_ref in zip(got.data, ref.data):
                        for a, b in zip(fam_got, fam_ref):
                            self.assertTrue(np.array_equal(a, b))

    def test_kronecker_equals_ragged(self):
        """The A-major ordering must match the ragged twin element-for-element -- a transposed pairing
        would still have the right ranks and the right multiset of values."""
        rng = np.random.default_rng(21)
        for struct in STRUCTURES:
            for pad in PADS:
                with self.subTest(struct=struct, pad=pad):
                    x = t3.TuckerTensorTrain.randn(*struct)
                    WA, WB = rand_ragged_weights(x, rng), rand_ragged_weights(x, rng)
                    _, uA = to_uniform_pair(x, WA, pad)
                    _, uB = to_uniform_pair(x, WB, pad)

                    got = uA.kronecker(uB).to_t3weights()
                    ref = WA.kronecker(WB)
                    for fam_got, fam_ref in zip(got.data, ref.data):
                        for a, b in zip(fam_got, fam_ref):
                            self.assertTrue(np.allclose(a, b))

    def test_concatenate_mask_is_exactly_the_concatenation(self):
        """Exact output masks, derived from the inputs' masks rather than read back from the impl -- and
        deliberately with SLACK on A, so the concatenation is gappy (a hole between A's real slots and B's
        block). Padded widths add."""
        rng = np.random.default_rng(22)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        _, uA = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))   # forced pad -> A has slack
        _, uB = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        z = uA.concatenate(uB)

        self.assertEqual(z.n, uA.n + uB.n)
        self.assertEqual(z.r, uA.r + uB.r)
        want_tk = np.concatenate([uA.masks.tucker_edge_mask, uB.masks.tucker_edge_mask], axis=-1)
        want_tt = np.concatenate([uA.masks.tt_edge_mask, uB.masks.tt_edge_mask], axis=-1)
        self.assertTrue(np.array_equal(z.masks.tucker_edge_mask, want_tk))
        self.assertTrue(np.array_equal(z.masks.tt_edge_mask, want_tt))

        # It really is gappy: A's slack sits between the two real blocks, so the mask is not a prefix.
        row = z.masks.tucker_edge_mask[0]
        self.assertFalse(np.array_equal(row, np.arange(len(row)) < row.sum()))   # NOT a prefix
        self.assertTrue(row[uA.n])                                               # B's block starts at nA

    def test_kronecker_mask_is_the_strided_outer_product(self):
        """The Kronecker mask against a hand-built expectation: real iff mask_A[a] AND mask_B[b] at the
        flattened index a*nB + b -- strided over the PADDED width, so gappy even from prefix inputs. This
        is the worked example in docs/uniform_masks_vs_ranks.md."""
        rng = np.random.default_rng(23)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        _, uA = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        _, uB = to_uniform_pair(x, rand_ragged_weights(x, rng), (3, 2))
        z = uA.kronecker(uB)

        self.assertEqual(z.n, uA.n * uB.n)
        self.assertEqual(z.r, uA.r * uB.r)

        # Hand-built from the definition (a double loop), not from the implementation's reshape.
        mA, mB = uA.masks.tucker_edge_mask, uB.masks.tucker_edge_mask
        want = np.zeros((mA.shape[0], uA.n * uB.n), dtype=bool)
        for mode in range(mA.shape[0]):
            for a in range(uA.n):
                for b in range(uB.n):
                    want[mode, a * uB.n + b] = mA[mode, a] and mB[mode, b]
        self.assertTrue(np.array_equal(z.masks.tucker_edge_mask, want))

        # Ranks multiply, and the pattern is strided-with-holes rather than a prefix.
        self.assertTrue(np.array_equal(np.asarray(z.tucker_ranks),
                                       np.asarray(uA.tucker_ranks) * np.asarray(uB.tucker_ranks)))
        row = z.masks.tucker_edge_mask[0]
        self.assertFalse(np.array_equal(row, np.arange(len(row)) < row.sum()))   # NOT a prefix

    def test_combine_masks_stay_host_numpy(self):
        """Masks are structure: they must stay host numpy through the combines even on jax supercores."""
        if not common.jax_available:
            self.skipTest('jax not available')
        import jax.numpy as jnp
        rng = np.random.default_rng(24)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        _, uA = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        jA = ut3.UT3Weights(jnp.asarray(uA.tucker_weight_supercore),
                            jnp.asarray(uA.tt_weight_supercore), uA.masks)
        for name in ('concatenate', 'kronecker'):
            with self.subTest(op=name):
                out = getattr(jA, name)(jA)
                self.assertTrue(common.is_numpy_ndarray(out.masks.tucker_edge_mask))
                self.assertTrue(common.is_numpy_ndarray(out.masks.tt_edge_mask))


class TestUT3WeightsFromSvd(unittest.TestCase):
    def test_from_ut3svd_matches_ragged(self):
        """The uniform singular-value weights equal the ragged ones on the real parts, and pair with the
        t3svd RESULT (which is what carries their masks)."""
        rng = np.random.default_rng(25)
        for struct in STRUCTURES:
            with self.subTest(struct=struct):
                x = t3.TuckerTensorTrain.randn(*struct)
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)

                W = ut3.UT3Weights.from_ut3svd(ux)
                xs, _, _ = ux.t3svd()
                self.assertTrue(W.is_consistent_with(xs))       # pairs with the t3svd result

                ref = t3.T3Weights.from_t3svd(x)
                for fam_got, fam_ref in zip(W.to_t3weights().data, ref.data):
                    for a, b in zip(fam_got, fam_ref):
                        self.assertTrue(np.allclose(np.abs(a), np.abs(b), atol=1e-10))

    def test_from_ut3svd_reciprocal_is_the_gk_metric(self):
        """The headline path end-to-end: sigmas -> reciprocal -> weighted norm, finite and matching
        ragged. This is the composition that motivated the reciprocal padding guard."""
        rng = np.random.default_rng(26)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        xs, _, _ = ux.t3svd()

        gk = ut3.UT3Weights.from_ut3svd(ux).reciprocal()
        self.assertTrue(np.isfinite(gk.tucker_weight_supercore).all())
        got = ut3.weighted_norm(xs, gk)
        self.assertTrue(np.isfinite(got))

        xr = xs.to_t3()
        ref = t3.weighted_norm(xr, t3.T3Weights.from_t3svd(x).reciprocal())
        self.assertLess(abs(got - ref), 1e-8 * (abs(ref) + 1))


class TestUT3WeightsDispatch(unittest.TestCase):
    def test_jit_absorb_and_norm(self):
        """jit the weighted ops with the masks closed over and only the supercores traced (the backend
        recipe). A stray np.* on a tracer would raise; a stray jnp.* on a mask would leak a tracer."""
        if not common.jax_available:
            self.skipTest('jax not available')
        import jax
        import jax.numpy as jnp
        import t3toolbox.backend.ut3_linalg as ut3_linalg

        rng = np.random.default_rng(13)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux, uW = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        ref = ut3.weighted_norm(ux, uW)

        shape, masks = ux.shape, ux.masks.data
        w_masks = uW.masks.data

        @jax.jit
        def norm(tk, tt, wtk, wtt):     # masks closed over as host constants
            return ut3_linalg.ut3_weighted_norm((tk, tt, shape, masks), (wtk, wtt, w_masks))

        got = norm(jnp.asarray(ux.tucker_supercore), jnp.asarray(ux.tt_supercore),
                   jnp.asarray(uW.tucker_weight_supercore), jnp.asarray(uW.tt_weight_supercore))
        self.assertTrue(np.allclose(np.asarray(got), ref, rtol=1e-5))   # jax defaults to float32

    def test_jit_frontend_pytree(self):
        """UT3Weights is a registered pytree: supercores are traced children, the mask holder is static
        value-hashed aux -- so jit over the frontend objects just works, and a rebuilt-but-identical
        weight is the SAME cache key (no recompile)."""
        if not common.jax_available:
            self.skipTest('jax not available')
        import jax

        rng = np.random.default_rng(14)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux, uW = to_uniform_pair(x, rand_ragged_weights(x, rng), (5, 4))
        ux, uW = ux.to_jax(), ut3.UT3Weights(*[common.to_jax(s) for s in uW.supercores], uW.masks)

        f = jax.jit(lambda a, w: ut3.weighted_norm(a, w))
        first = np.asarray(f(ux, uW))
        rebuilt = ut3.UT3Weights(uW.tucker_weight_supercore, uW.tt_weight_supercore,
                                 ut3.UT3Masks(*[m.copy() for m in uW.masks.data]))
        self.assertEqual(rebuilt.masks, uW.masks)                       # value-hashed: same key
        self.assertTrue(np.allclose(np.asarray(f(ux, rebuilt)), first))
        self.assertEqual(f._cache_size(), 1)                            # ...so exactly ONE compile


if __name__ == '__main__':
    unittest.main()


# ---------------------------------------------------------------------------------------------------
# UT3FrameWeights -- the TANGENT metric (S3).
# ---------------------------------------------------------------------------------------------------
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_frame_variations_format as ubvf
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.ufv_operations as ufv_operations
import t3toolbox.backend.utv_operations as utv_operations

CK_STACKS = [((), ()), ((), (2,)), ((2,), ()), ((2,), (3,)), ((2,), (2, 3))]   # (C, K)


def make_ragged_tangent(struct, C, K):
    x = t3.TuckerTensorTrain.randn(*struct, stack_shape=C)
    frame, _ = bvf.t3_orthogonal_representations(x)
    return t3m.COREWISE.randn(frame, stack_shape=K)   # variations stack = K + C


def rand_ragged_frame_weights(v, rng, stack_shape=None):
    """A random ragged T3FrameWeights at v's frame. FRAME-LIKE: stack C, not the variations' K + C."""
    V, H = v.variations.data
    ss = v.frame.stack_shape if stack_shape is None else stack_shape
    d = len(V)
    return bvf.T3FrameWeights(
        tuple(np.abs(rng.standard_normal(ss + (H[i].shape[-2],))) + 0.5 for i in range(d)),   # up   <- nU
        tuple(np.abs(rng.standard_normal(ss + (V[i].shape[-2],))) + 0.5 for i in range(d)),   # down <- nD
        tuple(np.abs(rng.standard_normal(ss + (H[i].shape[-3],))) + 0.5 for i in range(d)),   # left <- rL
        tuple(np.abs(rng.standard_normal(ss + (H[i].shape[-1],))) + 0.5 for i in range(d)))   # right<- rR


def to_uniform_tangent_pair(v, W):
    """(ragged tangent, ragged metric) -> (uniform tangent, uniform metric) padded consistently."""
    uv = ut3m.UT3Tangent.from_t3tangent(v)
    _, _, nU, nD, rL, rR, _ = uv.variations.uniform_structure
    return uv, ubvf.UT3FrameWeights.from_t3frameweights(W, nU=nU, nD=nD, rL=rL, rR=rR)


class TestUT3FrameWeights(unittest.TestCase):
    def test_weighted_norm_inner_equal_ragged(self):
        """The uniform tangent metric matches the ragged twin over structures x (C, K) stacks -- including
        a multi-axis K, where the C-over-K+C broadcast is easiest to get wrong."""
        rng = np.random.default_rng(30)
        for struct in STRUCTURES:
            for C, K in CK_STACKS:
                with self.subTest(struct=struct, C=C, K=K):
                    v = make_ragged_tangent(struct, C, K)
                    W = rand_ragged_frame_weights(v, rng)
                    uv, uW = to_uniform_tangent_pair(v, W)

                    self.assertEqual(uW.stack_shape, C)          # frame-like: C, never K + C
                    self.assertTrue(uW.is_consistent_with(uv))

                    got = np.asarray(uv.weighted_norm(uW))
                    ref = np.asarray(v.weighted_norm(W))
                    self.assertEqual(got.shape, K + C)
                    self.assertLess(np.abs(got - ref).max(), 1e-9 * (ref.max() + 1))

                    w2 = t3m.COREWISE.randn(v.frame, stack_shape=K)
                    uw2 = ut3m.UT3Tangent(uv.frame, to_uniform_tangent_pair(w2, W)[0].variations)
                    gi = np.asarray(uv.weighted_inner(uw2, uW))
                    ri = np.asarray(v.weighted_inner(w2, W))
                    self.assertLess(np.abs(gi - ri).max(), 1e-9 * (np.abs(ri).max() + 1))

    def test_C_weight_equals_K_tiled_weight(self):
        """The S3 design bet: one C-stacked metric shared by the K tangents at that frame gives exactly
        what an explicitly K-tiled metric would -- the leading '...' broadcast IS the tiling. (Reference
        via the backend, which is blind to the frame; the frontend rejects a K+C metric.)"""
        rng = np.random.default_rng(31)
        C, K = (2,), (3,)
        v = make_ragged_tangent(STRUCTURES[0], C, K)
        uv, uW = to_uniform_tangent_pair(v, rand_ragged_frame_weights(v, rng))

        def tile(a):
            return np.broadcast_to(a.reshape(a.shape[:1] + (1,) * len(K) + a.shape[1:]),
                                   a.shape[:1] + K + a.shape[1:]).copy()
        tiled = tuple(tile(a) for a in uW.supercores) + (tuple(tile(m) for m in uW.masks.data),)

        n_stack = len(uv.stack_shape)
        got = np.asarray(uv.weighted_norm(uW))
        ref = np.asarray(utv_operations.utv_weighted_norm(uv.variations.data, tiled, n_stack))
        self.assertTrue(np.array_equal(got, ref))    # exactly, not approximately

    def test_all_ones_metric_is_corewise_norm(self):
        """An all-ones metric must recover the plain coordinate norm exactly."""
        rng = np.random.default_rng(32)
        for C, K in [((), ()), ((2,), (3,))]:
            with self.subTest(C=C, K=K):
                v = make_ragged_tangent(STRUCTURES[0], C, K)
                uv, uW = to_uniform_tangent_pair(v, rand_ragged_frame_weights(v, rng))
                ones = ubvf.UT3FrameWeights(*[np.ones_like(a) for a in uW.supercores], uW.masks)
                self.assertLess(np.abs(np.asarray(uv.weighted_norm(ones))
                                       - np.asarray(uv.corewise_norm())).max(), 1e-12)

    def test_absorb_weights_and_gauge(self):
        """absorb_weights returns the weighted tangent at the SAME frame; its corewise_norm equals
        weighted_norm; and it is NOT gauged (weighting breaks the gauge, as in ragged)."""
        rng = np.random.default_rng(33)
        v = make_ragged_tangent(STRUCTURES[0], (), (2,))
        uv, uW = to_uniform_tangent_pair(v, rand_ragged_frame_weights(v, rng))
        gauged = ut3m.UNIFORM_MANIFOLD.project_oblique(uv)

        vw = gauged.absorb_weights(uW)
        self.assertIs(vw.frame, gauged.frame)                       # same frame object, untouched
        self.assertEqual(vw.variations.masks, gauged.variations.masks)   # absorb preserves the masks
        self.assertLess(np.abs(np.asarray(vw.corewise_norm())
                               - np.asarray(gauged.weighted_norm(uW))).max(), 1e-12)
        self.assertTrue(np.all(gauged.is_gauged()))
        self.assertFalse(np.all(vw.is_gauged()))                    # weighting breaks the gauge

    def test_roundtrip_and_from_ut3weights(self):
        """to_uniform -> to_ragged recovers the ragged metric; from_ut3weights slices base-point weights
        into the four families (up=down=tucker, left=tt[:-1], right=tt[1:])."""
        rng = np.random.default_rng(34)
        v = make_ragged_tangent(STRUCTURES[0], (), ())
        W = rand_ragged_frame_weights(v, rng)
        _, uW = to_uniform_tangent_pair(v, W)
        back = uW.to_t3frameweights()
        for fam_got, fam_ref in zip(back.data, W.data):
            for a, b in zip(fam_got, fam_ref):
                self.assertTrue(np.array_equal(a, b))

        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        uWb = ut3.UT3Weights.from_ut3svd(ux)
        fw = ubvf.UT3FrameWeights.from_ut3weights(uWb)
        self.assertTrue(np.array_equal(fw.up_weight_supercore, fw.down_weight_supercore))   # up == down
        self.assertTrue(np.array_equal(fw.left_weight_supercore, uWb.tt_weight_supercore[:-1]))
        self.assertTrue(np.array_equal(fw.right_weight_supercore, uWb.tt_weight_supercore[1:]))

        ref = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x))
        for fam_got, fam_ref in zip(fw.to_t3frameweights().data, ref.data):
            for a, b in zip(fam_got, fam_ref):
                self.assertTrue(np.allclose(np.abs(a), np.abs(b), atol=1e-10))

    def test_gk_metric_end_to_end(self):
        """The headline path: sigmas -> frame metric -> reciprocal -> weighted tangent norm. Finite (the
        padding guard) and matching ragged."""
        rng = np.random.default_rng(35)
        x = t3.TuckerTensorTrain.randn(*STRUCTURES[0])
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        frame, _ = ubvf.ut3_orthogonal_representations(ux)
        gk = ubvf.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(ux)).reciprocal()
        self.assertTrue(all(np.isfinite(a).all() for a in gk.supercores))

        v = ut3m.UNIFORM_COREWISE.randn(frame)          # unstacked: to_t3variations gives one object
        got = np.asarray(v.weighted_norm(gk))
        self.assertTrue(np.isfinite(got).all())

        rframe, _ = bvf.t3_orthogonal_representations(x)
        rgk = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x)).reciprocal()
        rv = t3m.T3Tangent(rframe, v.variations.to_t3variations())
        self.assertLess(np.abs(got - np.asarray(rv.weighted_norm(rgk))).max(), 1e-7 * (got.max() + 1))

    def test_reciprocal_padding_is_guarded(self):
        """Same hazard as the base-point layer, on all four families: naive 1/w on the padding gives inf;
        the guarded op stays finite with canonical zero padding."""
        rng = np.random.default_rng(36)
        v = make_ragged_tangent(STRUCTURES[0], (), ())
        uv, uW = to_uniform_tangent_pair(v, rand_ragged_frame_weights(v, rng))
        # Force padding to exist on every family.
        wide = ubvf.UT3FrameWeights(
            *[np.pad(a, ((0, 0),) * (a.ndim - 1) + ((0, 2),)) for a in uW.supercores],
            ubvf.UT3VariationsMasks(*[np.pad(m, ((0, 0),) * (m.ndim - 1) + ((0, 2),)) for m in uW.masks.data]))
        with np.errstate(divide='ignore'):
            self.assertTrue(np.isinf(1.0 / wide.up_weight_supercore).any())   # the hazard is real
        out = wide.reciprocal()
        for a, m in zip(out.supercores, wide.masks.data):
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue((a[~m] == 0).all())

    def test_frame_like_precondition(self):
        """check_ufw_pair rejects a metric whose stack is not the frame's C -- a K+C metric would weight
        one frame's K tangents with K DIFFERENT metrics. The blind backend predicate still accepts it
        (it reads as C_w = K+C), exactly as in the ragged layer: two levels, both deliberate."""
        rng = np.random.default_rng(37)
        C, K = (2,), (3,)
        v = make_ragged_tangent(STRUCTURES[0], C, K)
        uv, _ = to_uniform_tangent_pair(v, rand_ragged_frame_weights(v, rng))
        W_bad = rand_ragged_frame_weights(v, rng, stack_shape=K + C)     # NOT frame-like
        _, uW_bad = to_uniform_tangent_pair(v, W_bad)

        self.assertTrue(uW_bad.is_consistent_with(uv))                   # blind predicate accepts
        for name, op in (('weighted_norm', lambda: uv.weighted_norm(uW_bad)),
                         ('weighted_inner', lambda: uv.weighted_inner(uv, uW_bad)),
                         ('absorb_weights', lambda: uv.absorb_weights(uW_bad))):
            with self.subTest(op=name), self.assertRaises(ValueError):
                op()

    def test_mask_mismatch_is_rejected(self):
        """The uniform-only precondition, on each of the four families in turn: a metric declaring a
        different rank mask has the SAME padded shape, so nothing errors naturally -- absorbing it would
        silently zero a real variation slot.

        The mismatch must be mask-only: perturb the mask while leaving every supercore untouched, so the
        shape checks all pass and only the mask comparison can reject it. (An earlier version widened the
        supercores too, and so was rejected on shape -- it passed while the mask check was disabled.)"""
        rng = np.random.default_rng(38)
        v = make_ragged_tangent(STRUCTURES[0], (), ())
        uv, uW = to_uniform_tangent_pair(v, rand_ragged_frame_weights(v, rng))
        self.assertTrue(uW.is_consistent_with(uv))

        for fam, name in enumerate(('up', 'down', 'left', 'right')):
            masks = [m.copy() for m in uW.masks.data]
            last_real = int(masks[fam][0].sum()) - 1
            masks[fam][0, last_real] = False             # declare a REAL slot to be padding
            bad = ubvf.UT3FrameWeights(*uW.supercores, ubvf.UT3VariationsMasks(*masks))
            with self.subTest(family=name):
                # Every supercore is byte-identical to the good metric -- only the mask differs.
                for a, b in zip(bad.supercores, uW.supercores):
                    self.assertTrue(np.array_equal(a, b))
                self.assertFalse(bad.is_consistent_with(uv))
                for op_name, op in (('weighted_norm', lambda: uv.weighted_norm(bad)),
                                    ('absorb_weights', lambda: uv.absorb_weights(bad))):
                    with self.subTest(op=op_name), self.assertRaises(ValueError):
                        op()

    def test_masks_stay_host_numpy(self):
        """Masks are structure: host numpy even on jax supercores, through every op."""
        if not common.jax_available:
            self.skipTest('jax not available')
        import jax.numpy as jnp
        rng = np.random.default_rng(39)
        v = make_ragged_tangent(STRUCTURES[0], (), ())
        _, uW = to_uniform_tangent_pair(v, rand_ragged_frame_weights(v, rng))
        jW = ubvf.UT3FrameWeights(*[jnp.asarray(a) for a in uW.supercores], uW.masks)
        for name in ('reciprocal', 'sqrt'):
            with self.subTest(op=name):
                self.assertTrue(all(common.is_numpy_ndarray(m) for m in getattr(jW, name)().masks.data))
        for name in ('concatenate', 'kronecker'):
            with self.subTest(op=name):
                self.assertTrue(all(common.is_numpy_ndarray(m) for m in getattr(jW, name)(jW).masks.data))
