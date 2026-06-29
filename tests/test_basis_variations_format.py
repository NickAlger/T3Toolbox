# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
import numpy as np
import unittest

import t3toolbox.basis_variations_format as bvf
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.corewise as cw

np.random.seed(0)
tol = 1e-9
norm = np.linalg.norm
randn = np.random.randn


def _random_basis_variations(structure):
    """Build a consistent (T3Basis, T3Variations) pair from a rank spec.

    structure = (shape, up_ranks, down_ranks, left_ranks, right_ranks, stack_shape)
        up_ranks, down_ranks:  len=d
        left_ranks, right_ranks: len=d+1
    Core shapes (with leading stack_shape):
        U_i = (nU_i, N_i)           L_i = (rL_i, nU_i, rL_(i+1))
        R_i = (rR_i, nU_i, rR_(i+1))  D_i = (rL_i, nD_i, rR_(i+1))
        V_i = (nD_i, N_i)           H_i = (rL_i, nU_i, rR_(i+1))
    """
    shape, up_ranks, down_ranks, left_ranks, right_ranks, stack_shape = structure
    rnd = lambda *s: np.random.randn(*s)

    U = tuple(rnd(*(stack_shape + (nU, N)))       for nU, N       in zip(up_ranks, shape))
    L = tuple(rnd(*(stack_shape + (rL, nU, rLn))) for rL, nU, rLn in zip(left_ranks[:-1], up_ranks, left_ranks[1:]))
    R = tuple(rnd(*(stack_shape + (rR, nU, rRn))) for rR, nU, rRn in zip(right_ranks[:-1], up_ranks, right_ranks[1:]))
    D = tuple(rnd(*(stack_shape + (rL, nD, rRn))) for rL, nD, rRn in zip(left_ranks[:-1], down_ranks, right_ranks[1:]))
    V = tuple(rnd(*(stack_shape + (nD, N)))       for nD, N       in zip(down_ranks, shape))
    H = tuple(rnd(*(stack_shape + (rL, nU, rRn))) for rL, nU, rRn in zip(left_ranks[:-1], up_ranks, right_ranks[1:]))

    return bvf.T3Basis(U, D, L, R), bvf.T3Variations(V, H)


def _good_basis_cores():
    """A small, valid set of (up, down, left, right) cores for d=2, no stacking."""
    U = (randn(4, 14), randn(5, 15))            # (nU_i, N_i)
    L = (randn(1, 4, 2), randn(2, 5, 1))        # (rL_i, nU_i, rL_(i+1))
    R = (randn(1, 4, 2), randn(2, 5, 1))        # (rR_i, nU_i, rR_(i+1))
    D = (randn(1, 3, 2), randn(2, 4, 1))        # (rL_i, nD_i, rR_(i+1))
    return U, D, L, R


def _slice_basis(base, idx):
    """The unstacked T3Basis at stack index ``idx`` (idx=() returns the whole base)."""
    s = lambda C: np.asarray(C)[idx]
    up, down, left, right = base.data
    return bvf.T3Basis(tuple(map(s, up)), tuple(map(s, down)), tuple(map(s, left)), tuple(map(s, right)))


def _slice_variations(var, idx):
    """The unstacked T3Variations at stack index ``idx``."""
    s = lambda C: np.asarray(C)[idx]
    return bvf.T3Variations(tuple(map(s, var.tucker_variations)), tuple(map(s, var.tt_variations)))


class TestBasisVariationsFormat(unittest.TestCase):
    base_structures = [
        #  (shape,             up_ranks,      down_ranks,    left_ranks,        right_ranks)
        ((14,),                (4,),          (3,),          (1, 1),            (1, 1)),
        ((14, 15),             (4, 5),        (3, 4),        (1, 2, 1),         (1, 2, 1)),
        ((14, 15, 16),         (4, 5, 6),     (3, 4, 5),     (1, 2, 3, 1),      (1, 3, 2, 1)),
        ((10, 11, 12, 13),     (2, 3, 4, 3),  (2, 2, 3, 2),  (1, 2, 3, 2, 1),   (1, 2, 2, 3, 1)),
    ]
    stack_shapes = [(), (2,), (2, 3)]

    def _equal_cores(self, AA, BB):
        self.assertEqual(len(AA), len(BB))
        for A, B in zip(AA, BB):
            self.assertTrue(np.array_equal(np.asarray(A), np.asarray(B)))

    @staticmethod
    def _expected_variation_shapes(structure):
        shape, up_ranks, down_ranks, left_ranks, right_ranks = structure[:5]
        tucker_variation_shapes = tuple(zip(down_ranks, shape))
        tt_variation_shapes = tuple(zip(left_ranks[:-1], up_ranks, right_ranks[1:]))
        return tucker_variation_shapes, tt_variation_shapes

    def test_t3basis_properties(self):
        for BASE_STRUCTURE in self.base_structures:
            shape, up_ranks, down_ranks, left_ranks, right_ranks = BASE_STRUCTURE
            for STACK_SHAPE in self.stack_shapes:
                structure = BASE_STRUCTURE + (STACK_SHAPE,)
                with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    base, _ = _random_basis_variations(structure)

                    self.assertEqual(len(shape), base.d)
                    self.assertEqual(shape, base.shape)
                    self.assertEqual(up_ranks, base.up_ranks)
                    self.assertEqual(down_ranks, base.down_ranks)
                    self.assertEqual(left_ranks, base.left_ranks)
                    self.assertEqual(right_ranks, base.right_ranks)
                    self.assertEqual(STACK_SHAPE, base.stack_shape)
                    self.assertEqual(structure, base.structure)
                    self.assertEqual(self._expected_variation_shapes(structure), base.variation_shapes)
                    self.assertEqual((base.up_tucker_cores, base.down_tt_cores,
                                      base.left_tt_cores, base.right_tt_cores), base.data)

    def test_t3variations_properties(self):
        for BASE_STRUCTURE in self.base_structures:
            shape, up_ranks, down_ranks, left_ranks, right_ranks = BASE_STRUCTURE
            for STACK_SHAPE in self.stack_shapes:
                structure = BASE_STRUCTURE + (STACK_SHAPE,)
                with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    _, variations = _random_basis_variations(structure)

                    self.assertEqual(len(shape), variations.d)
                    self.assertEqual(shape, variations.shape)
                    self.assertEqual(STACK_SHAPE, variations.stack_shape)
                    self.assertEqual(self._expected_variation_shapes(structure), variations.variation_shapes)
                    self.assertEqual((variations.tucker_variations, variations.tt_variations), variations.data)

    def test_t3basis_validate_raises(self):
        # Each corruption introduces exactly one inconsistency into an otherwise-valid set of cores.
        corruptions = [
            ("wrong number of cores",   lambda U, D, L, R: (U[:-1], D, L, R)),
            ("up core not a matrix",    lambda U, D, L, R: ((randn(4),) + U[1:], D, L, R)),
            ("left core not 3-tensor",  lambda U, D, L, R: (U, D, (randn(1, 4),) + L[1:], R)),
            ("tucker rank mismatch",    lambda U, D, L, R: ((randn(99, 14),) + U[1:], D, L, R)),
            ("down-left rank mismatch", lambda U, D, L, R: (U, (randn(99, 3, 2),) + D[1:], L, R)),
            ("down-right rank mismatch", lambda U, D, L, R: (U, (randn(1, 3, 99),) + D[1:], L, R)),
            ("left chain inconsistent", lambda U, D, L, R: (U, D, (L[0], randn(99, 5, 1)), R)),
            ("right chain inconsistent", lambda U, D, L, R: (U, D, L, (R[0], randn(99, 5, 1)))),
            ("inconsistent stack shape", lambda U, D, L, R: (U, D, (randn(2, 1, 4, 2),) + L[1:], R)),
        ]
        for label, corrupt in corruptions:
            with self.subTest(corruption=label):
                U, D, L, R = _good_basis_cores()
                with self.assertRaises(ValueError):
                    bvf.T3Basis(*corrupt(U, D, L, R))

    def test_t3basis_validate_accepts_good(self):
        U, D, L, R = _good_basis_cores()
        bvf.T3Basis(U, D, L, R)  # must not raise

    def test_t3variations_validate_raises(self):
        def good():
            V = (randn(3, 14), randn(4, 15))         # (nD_i, N_i)
            H = (randn(1, 4, 2), randn(2, 5, 1))     # (rL_i, nU_i, rR_(i+1))
            return V, H

        corruptions = [
            ("wrong number of cores",    lambda V, H: (V[:-1], H)),
            ("tucker variation not matrix", lambda V, H: ((randn(3),) + V[1:], H)),
            ("tt variation not 3-tensor", lambda V, H: (V, (randn(1, 4),) + H[1:])),
            ("inconsistent stack shape", lambda V, H: ((randn(2, 3, 14),) + V[1:], H)),
        ]
        for label, corrupt in corruptions:
            with self.subTest(corruption=label):
                V, H = good()
                with self.assertRaises(ValueError):
                    bvf.T3Variations(*corrupt(V, H))

    def test_check_bv_pair(self):
        structure = ((14, 15, 16), (4, 5, 6), (3, 4, 5), (1, 2, 3, 1), (1, 3, 2, 1), ())
        base, variations = _random_basis_variations(structure)
        bvf.check_bv_pair(base, variations)  # consistent: must not raise

        V, H = variations.tucker_variations, variations.tt_variations

        # Tucker variation does not fit a base hole
        badV = (randn(V[0].shape[-2] + 1, V[0].shape[-1]),) + V[1:]
        with self.assertRaises(ValueError):
            bvf.check_bv_pair(base, bvf.T3Variations(badV, H))

        # TT variation does not fit a base hole
        badH = (randn(H[0].shape[-3] + 1, H[0].shape[-2], H[0].shape[-1]),) + H[1:]
        with self.assertRaises(ValueError):
            bvf.check_bv_pair(base, bvf.T3Variations(V, badH))

    def test_check_bv_pair_stacking(self):
        # A variation may carry extra OUTER tangent-stack axes V; consistency requires the base
        # stack G to be the trailing (inner) part of the variation stack (variation stack = V + G).
        def variations_with_stack(base, full_stack):
            VV_shapes, HH_shapes = base.variation_shapes
            VV = tuple(randn(*(full_stack + s)) for s in VV_shapes)
            HH = tuple(randn(*(full_stack + s)) for s in HH_shapes)
            return bvf.T3Variations(VV, HH)

        STRUCT = ((14, 15, 16), (4, 5, 6), (3, 4, 5), (1, 2, 3, 1), (1, 3, 2, 1))

        # base stack G is a suffix of the variation stack -> consistent (incl. extra tangent stack V)
        for BASE_STACK, TANGENT_STACK in [((), ()), ((), (4,)), ((2,), ()), ((2,), (4,)), ((2, 3), (4,))]:
            with self.subTest(case="ok", BASE_STACK=BASE_STACK, TANGENT_STACK=TANGENT_STACK):
                base, _ = _random_basis_variations(STRUCT + (BASE_STACK,))
                bvf.check_bv_pair(base, variations_with_stack(base, TANGENT_STACK + BASE_STACK))

        # base stack G is NOT a suffix of the variation stack -> raises
        for BASE_STACK, VAR_STACK in [((2,), (3,)), ((2,), ()), ((2, 3), (2,)), ((2, 3), (4, 2))]:
            with self.subTest(case="bad", BASE_STACK=BASE_STACK, VAR_STACK=VAR_STACK):
                base, _ = _random_basis_variations(STRUCT + (BASE_STACK,))
                with self.assertRaises(ValueError):
                    bvf.check_bv_pair(base, variations_with_stack(base, VAR_STACK))

    def test_stack_unstack(self):
        for BASE_STRUCTURE in self.base_structures:
            for STACK_SHAPE in [(2,), (2, 3)]:
                structure = BASE_STRUCTURE + (STACK_SHAPE,)
                with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    base, variations = _random_basis_variations(structure)

                    # round trips
                    base2 = bvf.T3Basis.stack(base.unstack())
                    self.assertLessEqual(float(cw.corewise_norm(cw.corewise_sub(base.data, base2.data))), tol)
                    variations2 = bvf.T3Variations.stack(variations.unstack())
                    self.assertLessEqual(float(cw.corewise_norm(cw.corewise_sub(variations.data, variations2.data))), tol)

                    # an unstacked leaf equals the manually sliced sub-basis
                    idx = tuple(0 for _ in STACK_SHAPE)
                    leaf = base.unstack()
                    for k in idx:
                        leaf = leaf[k]
                    sliced = bvf.T3Basis(
                        tuple(U[idx] for U in base.up_tucker_cores),
                        tuple(G[idx] for G in base.down_tt_cores),
                        tuple(G[idx] for G in base.left_tt_cores),
                        tuple(G[idx] for G in base.right_tt_cores),
                    )
                    self.assertLessEqual(float(cw.corewise_norm(cw.corewise_sub(sliced.data, leaf.data))), tol)

    def test_bv_to_t3(self):
        for BASE_STRUCTURE in self.base_structures:
            shape = BASE_STRUCTURE[0]
            d = len(shape)
            for STACK_SHAPE in self.stack_shapes:
                structure = BASE_STRUCTURE + (STACK_SHAPE,)
                base, variations = _random_basis_variations(structure)
                U, D, L, R = base.data
                V, H = variations.data
                for ii in range(d):
                    with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                      ii=ii, kind="TT"):
                        x = bvf.bv_to_t3((True, ii), base, variations)
                        self._equal_cores(x.tucker_cores, U)
                        self._equal_cores(x.tt_cores, L[:ii] + (H[ii],) + R[ii + 1:])

                    with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, STACK_SHAPE=STACK_SHAPE,
                                      ii=ii, kind="Tucker"):
                        x = bvf.bv_to_t3((False, ii), base, variations)
                        self._equal_cores(x.tucker_cores, U[:ii] + (V[ii],) + U[ii + 1:])
                        self._equal_cores(x.tt_cores, L[:ii] + (D[ii],) + R[ii + 1:])

    def test_bv_to_t3_tangent_stacked(self):
        # A V-stacked variation (tangent stack V over base stack G): bv_to_t3 must broadcast the
        # G-stacked base cores up to V+G so the term is a valid uniform-stack TuckerTensorTrain whose
        # every (v, g) slice equals the corresponding unstacked term (base shared across V).
        for BASE_STRUCTURE in self.base_structures:
            shape = BASE_STRUCTURE[0]
            d = len(shape)
            for BASE_G, V in [((), (2,)), ((2,), (2,))]:
                with self.subTest(BASE_STRUCTURE=BASE_STRUCTURE, BASE_G=BASE_G, V=V):
                    base, _ = _random_basis_variations(BASE_STRUCTURE + (BASE_G,))
                    rnd = lambda *s: np.random.randn(*s)
                    VG = V + BASE_G
                    tuck_shapes, tt_shapes = base.variation_shapes
                    var = bvf.T3Variations(
                        tuple(rnd(*(VG + s)) for s in tuck_shapes),
                        tuple(rnd(*(VG + s)) for s in tt_shapes),
                    )
                    for ii in range(d):
                        for use_tt in (True, False):
                            term = bvf.bv_to_t3((use_tt, ii), base, var)
                            self.assertEqual(VG, term.stack_shape)  # valid uniform-stack T3
                            term_dense = np.asarray(term.to_dense())
                            for idx in np.ndindex(*VG):
                                g_idx = idx[len(idx) - len(BASE_G):] if BASE_G else ()
                                ref = bvf.bv_to_t3((use_tt, ii), _slice_basis(base, g_idx),
                                                   _slice_variations(var, idx))
                                self.check_relerr(np.asarray(ref.to_dense()), term_dense[idx])

    # ----------------------------------------------------------------------
    # Reconstruction / orthogonality tier: t3_orthogonal_representations
    # ----------------------------------------------------------------------

    t3_structures = [
        #  (shape,             tucker_ranks,   tt_ranks)
        ((14,),                (4,),           (1, 1)),
        ((14, 15),             (4, 5),         (1, 3, 1)),
        ((14, 15, 16),         (4, 5, 6),      (1, 3, 2, 1)),
        ((14, 15, 16, 17),     (4, 5, 6, 5),   (1, 3, 4, 2, 1)),
    ]

    def check_relerr(self, xtrue, x):
        xtrue, x = np.asarray(xtrue), np.asarray(x)
        self.assertLessEqual(norm(xtrue - x), tol * norm(xtrue))

    def test_metadata_repr_backend(self):
        # T3Basis / T3Variations slice-1: size/data_size, minimal_ranks, backend-convert, copy, repr.
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        for C in [(), (2,)]:
            x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
            base, var = bvf.t3_orthogonal_representations(x)
            for obj in (base, var):
                self.assertEqual(int(np.prod(STRUCT[0])), obj.size)            # dense element count
                self.assertEqual(sum(int(c.size) for fam in obj.data for c in fam), obj.data_size)
                self.assertFalse(obj.contains_jax)
                cp = obj.copy(); cp.data[0][0][...] = 7.0                       # copy is independent
                self.assertFalse(np.allclose(np.asarray(obj.data[0][0]), 7.0))
                self.assertIn(type(obj).__name__, repr(obj))                   # concise repr (no array dump)
                self.assertNotIn("array", repr(obj))
            self.assertEqual(                                                  # structural minimal ranks
                t3.TuckerTensorTrain.get_minimal_ranks(base.shape, base.up_ranks, base.left_ranks),
                base.minimal_ranks)
            try:
                import jax  # noqa: F401
                self.assertTrue(base.to_jax().contains_jax and var.to_jax().contains_jax)
                self.assertFalse(base.to_jax().to_numpy().contains_jax)
            except ImportError:
                pass

    def test_constructors(self):
        # T3Basis.from_t3 / random_orthogonal / random_orthogonal_like; T3Variations zeros/randn/unit/_like.
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=(2,))
        b = bvf.T3Basis.from_t3(x)
        b2, _ = bvf.t3_orthogonal_representations(x)
        self.assertEqual(b.structure, b2.structure)
        self.assertTrue(b.is_orthogonal().all())
        ro = bvf.T3Basis.random_orthogonal(*STRUCT, stack_shape=(2,))
        self.assertTrue(ro.is_orthogonal().all())
        self.assertEqual((STRUCT[0], STRUCT[1], STRUCT[2], (2,)),
                         (ro.shape, ro.up_ranks, ro.left_ranks, ro.stack_shape))
        self.assertEqual(b.structure, bvf.T3Basis.random_orthogonal_like(b).structure)
        vs = b.variation_shapes
        z = bvf.T3Variations.zeros(vs, stack_shape=(2,))
        self.assertTrue(all(np.all(np.asarray(c) == 0) for c in z.tucker_variations + z.tt_variations))
        self.assertEqual((5, 6, 4), bvf.T3Variations.randn(vs, stack_shape=(2,)).shape)
        u = bvf.T3Variations.unit(vs, (True, 1, (0, 1, 0)), stack_shape=(2,))   # tt core 1, entry (0,1,0)
        self.assertEqual(2, sum(int(np.count_nonzero(np.asarray(c)))            # one '1' per stack element
                                for c in u.tucker_variations + u.tt_variations))
        self.assertEqual(b.stack_shape, bvf.T3Variations.zeros_like(b).stack_shape)
        self.assertEqual(z.stack_shape, bvf.T3Variations.randn_like(z).stack_shape)

    def test_to_from_vector(self):
        # T3Variations.to_vector / from_vector round-trip (flat length == stored DOF).
        base = bvf.T3Basis.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        var = bvf.T3Variations.randn(base.variation_shapes, stack_shape=(3, 2))
        flat = var.to_vector()
        self.assertEqual((var.data_size,), flat.shape)
        var2 = bvf.T3Variations.from_vector(flat, base.variation_shapes, stack_shape=(3, 2))
        self.assertEqual(0.0, cw.corewise_relerr(var.data, var2.data))

    def test_save_load(self):
        import tempfile, os
        base = bvf.T3Basis.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
        var = bvf.T3Variations.randn(base.variation_shapes, stack_shape=(2,))
        d = tempfile.mkdtemp()
        for obj, loader, name in [(base, bvf.T3Basis.load, 'b'), (var, bvf.T3Variations.load, 'v')]:
            f = os.path.join(d, name + '.npz'); obj.save(f)
            self.assertEqual(0.0, cw.corewise_relerr(obj.data, loader(f).data))

    def test_reverse(self):
        # T3Basis.reverse stays orthogonal with reversed shape; reverse is an involution.
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        base = bvf.T3Basis.random_orthogonal(*STRUCT, stack_shape=(2,))
        rb = base.reverse()
        self.assertEqual(STRUCT[0][::-1], rb.shape)
        self.assertTrue(rb.is_orthogonal().all())
        self.assertEqual(base.structure, rb.reverse().structure)
        var = bvf.T3Variations.randn(base.variation_shapes, stack_shape=(2,))
        self.assertEqual(0.0, cw.corewise_relerr(var.data, var.reverse().reverse().data))

    def test_variations_arithmetic(self):
        # T3Variations corewise +,-,*,neg correspond to tangent linearity; sum_stack reduces the stack.
        import t3toolbox.manifold as t3m
        base = bvf.T3Basis.random_orthogonal((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        a = t3m.COREWISE.randn(base).variations
        b = t3m.COREWISE.randn(base).variations
        dn = lambda var: np.asarray(t3m.T3Tangent(base, var).to_dense())
        self.check_relerr(dn(a) + dn(b), dn(a + b))
        self.check_relerr(dn(a) - dn(b), dn(a - b))
        self.check_relerr(2.5 * dn(a), dn(2.5 * a))
        self.check_relerr(-dn(a), dn(-a))
        vs = bvf.T3Variations.randn(base.variation_shapes, stack_shape=(3,))
        self.assertEqual((), vs.sum_stack().stack_shape)

    def test_to_t3_to_dense(self):
        # T3Basis.to_t3 / to_dense reconstruct the base point (== the original T3 for a consistent basis).
        STRUCT = ((5, 6, 4), (2, 3, 2), (1, 2, 2, 1))
        for C in [(), (2,)]:
            x = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=C)
            base = bvf.T3Basis.from_t3(x)
            self.check_relerr(np.asarray(x.to_dense()), np.asarray(base.to_dense()))
            self.check_relerr(np.asarray(x.to_dense()), np.asarray(base.to_t3().to_dense()))
            self.assertEqual(x.shape, base.to_t3().shape)

    def test_orthogonalize_is_consistent(self):
        # orthogonalize() reconstructs the same base point as a valid orthogonal basis; is_consistent()
        # is True for a from_t3 basis and False when the left/right reconstructions disagree.
        for C in [(), (2,)]:
            x = t3.TuckerTensorTrain.randn((5, 6, 4), (3, 4, 3), (1, 2, 3, 1), stack_shape=C)
            base = bvf.T3Basis.from_t3(x)
            b2 = base.orthogonalize()
            b2.validate()
            self.assertTrue(b2.is_orthogonal().all())
            self.check_relerr(np.asarray(x.to_dense()), np.asarray(b2.to_dense()))
            self.assertTrue(base.is_consistent().all())
            # perturb the left cores only -> left/right reconstructions no longer agree
            bad = bvf.T3Basis(base.up_tucker_cores, base.down_tt_cores,
                              tuple(c + 0.1 * np.random.randn(*c.shape) for c in base.left_tt_cores),
                              base.right_tt_cores)
            self.assertFalse(bad.is_consistent().all())

    def test_allclose(self):
        # T3Basis.allclose compares represented base points (gauge-invariant); T3Variations.allclose
        # compares variations corewise.
        for C in [(), (2,)]:
            x = t3.TuckerTensorTrain.randn((5, 6, 4), (3, 4, 3), (1, 2, 3, 1), stack_shape=C)
            base = bvf.T3Basis.from_t3(x)
            self.assertTrue(base.allclose(base).all())
            self.assertTrue(base.allclose(base.orthogonalize()).all())  # same point, possibly different gauge
            y = t3.TuckerTensorTrain.randn((5, 6, 4), (3, 4, 3), (1, 2, 3, 1), stack_shape=C)
            self.assertFalse(base.allclose(bvf.T3Basis.from_t3(y)).all())

            _, variations = bvf.t3_orthogonal_representations(x)
            self.assertTrue(variations.allclose(variations).all())
            self.assertFalse(variations.allclose(variations * 2.0).all())
            self.assertTrue(variations.allclose(variations + variations * 1e-12).all())

    def _assert_orthonormal(self, gram, n):
        # gram has shape stack_shape + (n, n); each stacked block must be the identity
        self.assertLessEqual(norm(np.asarray(gram) - np.eye(n)), tol)

    def test_orthogonal_representations_reconstruction(self):
        # Replacing any single hole with its matching variation reconstructs the original tensor.
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in self.stack_shapes:
                x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    base, variations = bvf.t3_orthogonal_representations(x)
                    x_dense = x.to_dense()
                    for ii in range(x.d):
                        self.check_relerr(x_dense, bvf.bv_to_t3((True, ii), base, variations).to_dense())
                        self.check_relerr(x_dense, bvf.bv_to_t3((False, ii), base, variations).to_dense())

    def test_orthogonal_representations_base_orthogonality(self):
        # U: up-orthogonal (all i); D: outer-orthogonal (all i);
        # L: left-orthogonal (i=0..d-2); R: right-orthogonal (i=1..d-1).
        for T3_STRUCTURE in self.t3_structures:
            for STACK_SHAPE in self.stack_shapes:
                x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    base, _ = bvf.t3_orthogonal_representations(x)
                    U = [np.asarray(c) for c in base.up_tucker_cores]
                    D = [np.asarray(c) for c in base.down_tt_cores]
                    L = [np.asarray(c) for c in base.left_tt_cores]
                    R = [np.asarray(c) for c in base.right_tt_cores]
                    d = x.d

                    for ii in range(d):
                        self._assert_orthonormal(np.einsum('...io,...jo->...ij', U[ii], U[ii]), U[ii].shape[-2])
                        self._assert_orthonormal(np.einsum('...iaj,...ibj->...ab', D[ii], D[ii]), D[ii].shape[-2])
                    for ii in range(d - 1):
                        self._assert_orthonormal(np.einsum('...iaj,...iak->...jk', L[ii], L[ii]), L[ii].shape[-1])
                    for ii in range(1, d):
                        self._assert_orthonormal(np.einsum('...iaj,...kaj->...ik', R[ii], R[ii]), R[ii].shape[-3])

    def test_t3basis_is_orthogonal(self):
        # bases from t3_orthogonal_representations are orthogonal
        for T3_STRUCTURE in [((10, 11, 12), (3, 4, 3), (1, 2, 2, 1)),
                             ((9, 10, 11, 12), (2, 3, 3, 2), (1, 2, 3, 2, 1))]:
            for STACK_SHAPE in [(), (2,)]:
                with self.subTest(T3_STRUCTURE=T3_STRUCTURE, STACK_SHAPE=STACK_SHAPE):
                    x = t3.TuckerTensorTrain.randn(*T3_STRUCTURE, stack_shape=STACK_SHAPE)
                    base, _ = bvf.t3_orthogonal_representations(x)
                    self.assertTrue(base.is_orthogonal().all())

        # a generic (non-orthogonal) basis is not orthogonal
        structure = ((14, 15, 16), (4, 5, 6), (3, 4, 5), (1, 2, 3, 1), (1, 3, 2, 1), ())
        base2, _ = _random_basis_variations(structure)
        self.assertFalse(base2.is_orthogonal().all())

    def test_t3basis_has_minimal_ranks(self):
        # minimal-rank x -> minimal-rank base
        for STACK_SHAPE in [(), (2,), (2, 3)]:
            x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1), stack_shape=STACK_SHAPE)
            self.assertTrue(x.has_minimal_ranks)
            base, _ = bvf.t3_orthogonal_representations(x)
            self.assertTrue(base.has_minimal_ranks)

        # non-minimal Tucker rank (4 > rL*rR = 1*3) -> up_ranks != down_ranks -> not minimal
        x2 = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
        base2, _ = bvf.t3_orthogonal_representations(x2)
        self.assertFalse(base2.has_minimal_ranks)

        # hand-built basis with left_ranks != right_ranks and up_ranks != down_ranks -> not minimal
        structure = ((14, 15, 16), (4, 5, 6), (3, 4, 5), (1, 2, 3, 1), (1, 3, 2, 1), ())
        base3, _ = _random_basis_variations(structure)
        self.assertFalse(base3.has_minimal_ranks)

    def test_t3basis_has_numerically_minimal_ranks(self):
        # frame numerical minimality is certified WITHOUT an SVD: orthogonal AND structurally minimal.
        base, _ = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1)))     # orthogonal + minimal
        self.assertTrue(base.is_orthogonal().all() and base.has_minimal_ranks)
        self.assertTrue(base.has_numerically_minimal_ranks().all())

        nb, _ = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn((10, 11, 12), (4, 5, 4), (1, 2, 3, 1)))  # orthogonal, NON-minimal
        self.assertTrue(nb.is_orthogonal().all())
        self.assertFalse(nb.has_minimal_ranks)
        self.assertFalse(nb.has_numerically_minimal_ranks().all())                   # structural fail -> False

        # a non-orthogonal frame returns False (the SVD certification path is intentionally not built),
        # even when its ranks happen to be structurally minimal
        nonorth, _ = _random_basis_variations(((6, 7, 5), (2, 2, 2), (2, 2, 2), (1, 2, 2, 1), (1, 2, 2, 1), ()))
        self.assertFalse(nonorth.is_orthogonal().all())
        self.assertFalse(nonorth.has_numerically_minimal_ranks().all())


if __name__ == "__main__":
    unittest.main()
