# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The backend checkers of a frame (graded messages, P4): ``fv_frame_consistency_residual``,
``fv_centers_consistency_residual`` and their uniform twins ``ufv_frame_consistency_residual``,
``ufv_centers_consistency_residual``.

What is pinned:

- **the definition**: on hand-made frames and centers (random cores, so nothing is orthogonal or consistent) each
  residual equals its definition computed from DENSE tensors by this file's own einsum-free route
  (``t3_to_dense`` of each network), with stacks ``C`` and with tails;
- **the verdicts**: a frame the library builds passes at ``1e-12``, whatever the norm of its tensor (the difference
  of two inner products would stop near ``1e-8``; a difference taken across two gauges grows with the norm); a
  perturbed store fails; wrong centers fail, at every site and in either family; only the stack elements that are
  wrong fail;
- **no dense tensor**: a frame whose dense tensor has ``100**12`` entries is checked (it could not be allocated);
- **no overflow**: a float32 frame of norm ``1e20`` (whose squared norm is not representable) is judged like any
  other, and where a norm is not representable the residual is ``nan``, never 0;
- **the uniform twins**: equal to the ragged functions element by element on frames whose ranks vary over the
  stack, with tails, and unchanged by garbage in the padding of the frame and of the centers;
- **checkers, not guarantees**: a residual is returned, never an error, whatever the numbers; nothing else in the
  backend calls a checker.
"""
import ast
import pathlib
import unittest
import unittest.mock as mock
import warnings
import numpy as np

import t3toolbox
import t3toolbox.backend.t3_constructors as t3_constructors
import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.ufv_conversions as ufv_conversions
import t3toolbox.backend.ufv_masking as ufv_masking
import t3toolbox.backend.ufv_operations as ufvo
import t3toolbox.backend.ut3_masking as ut3_masking

# (shape, tucker_ranks, tt_ranks): squashed tails and open tails, d = 1, over-ranked Tucker cores, a long chain
_STRUCTS = [
    ((5, 6, 7), (3, 4, 2), (1, 3, 2, 1)),
    ((5, 6, 7), (3, 4, 2), (2, 3, 2, 3)),          # tails at both ends
    ((4,), (3,), (1, 1)),                          # d = 1
    ((4,), (3,), (2, 3)),                          # d = 1 with tails
    ((3, 4, 3, 5), (4, 2, 3, 2), (3, 2, 4, 2, 2)), # Tucker rank above the mode size (4 > 3), tails
    ((2, 9), (5, 3), (1, 6, 1)),
]
_STACKS = [(), (3,), (2, 2)]


# ---------------------------------------------------------------------------------- the definitions, densely
def _dense(tucker_cores, tt_cores):   # C + (N0, ..., N(d-1)); a tail is summed over its index
    return t3_conversions.t3_to_dense((tuple(tucker_cores), tuple(tt_cores)))


def _norm_over_modes(T, n_stack):     # shape = C
    return np.sqrt(np.sum(T ** 2, axis=tuple(range(n_stack, T.ndim))))


def dense_frame_residual(frame, n_stack):
    """``||T(D, L) - T(D, R)|| / max(1, ||T(D, R)||)`` from the two dense tensors."""
    D, _U, L, R = frame
    through_left, through_right = _dense(D, L), _dense(D, R)
    return (_norm_over_modes(through_left - through_right, n_stack)
            / np.maximum(1.0, _norm_over_modes(through_right, n_stack)))


def dense_centers_residuals(frame, centers, n_stack):
    """The centers' mismatch site by site and family by family, from dense tensors: a list of ``(family, site,
    residual)``, the residual of shape ``C``."""
    D, U, L, R = (tuple(family) for family in frame)
    Vc, Hc = centers
    base_point = _dense(D, L)
    den = np.maximum(1.0, _norm_over_modes(base_point, n_stack))
    out = []
    for ii in range(len(D)):
        if Vc is not None:    # Vc[i] under U[i]; L over D to the left, R over D to the right
            with_Vc = _dense(D[:ii] + (Vc[ii],) + D[ii + 1:], L[:ii] + (U[ii],) + R[ii + 1:])
            out.append(('V', ii, _norm_over_modes(with_Vc - base_point, n_stack) / den))
        if Hc is not None:    # Hc[i] over D[i]; L over D to the left, R over D to the right
            with_Hc = _dense(D, L[:ii] + (Hc[ii],) + R[ii + 1:])
            out.append(('H', ii, _norm_over_modes(with_Hc - base_point, n_stack) / den))
    return out


def dense_centers_residual(frame, centers, n_stack):
    return np.max(np.stack([res for _, _, res in dense_centers_residuals(frame, centers, n_stack)]), axis=0)


# ---------------------------------------------------------------------------------- fixtures
def library_frame(struct, C=(), squash_tails=True, scale=1.0, dtype=np.float64):
    """A frame and its centers as the library's orthogonalization returns them, the tensor scaled by ``scale``."""
    tucker_cores, tt_cores = t3_constructors.t3_corewise_randn(*struct, stack_shape=C)
    tucker_cores = (scale * tucker_cores[0],) + tuple(tucker_cores[1:])
    frame, centers = fv_conversions.t3_orthogonal_representations((tucker_cores, tt_cores), squash_tails=squash_tails)
    cast = lambda families: tuple(tuple(np.asarray(core, dtype=dtype) for core in family) for family in families)
    return cast(frame), cast(centers)


def random_like(families, rng):
    """Independent Gaussian arrays of the shapes of ``families``: a hand-made frame or hand-made centers."""
    return tuple(tuple(rng.standard_normal(core.shape) for core in family) for family in families)


def replace_core(frame, family, site, new_core):
    return tuple(tuple(new_core if (ff, ii) == (family, site) else core for ii, core in enumerate(cores))
                 for ff, cores in enumerate(frame))


def relative_difference(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))))


class TestRaggedFrameResidual(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.rng = np.random.default_rng(0)

    def test_it_is_the_definition_on_hand_made_frames(self):
        """Random cores of a frame's shapes, nothing orthogonal, nothing consistent: the residual through the
        cores is the residual of the two dense tensors, per stack element, with tails summed."""
        for struct in _STRUCTS:
            for C in _STACKS:
                for squash_tails in (True, False):
                    with self.subTest(struct=struct, C=C, squash_tails=squash_tails):
                        frame, _ = library_frame(struct, C, squash_tails)
                        hand_made = random_like(frame, self.rng)
                        residual = fvo.fv_frame_consistency_residual(hand_made)
                        self.assertEqual(residual.shape, C)
                        expected = dense_frame_residual(hand_made, len(C))
                        self.assertGreater(float(np.min(expected)), 1e-3)          # the case is not vacuous
                        self.assertLess(relative_difference(residual, expected), 1e-12)

    def test_below_norm_one_the_residual_is_absolute(self):
        """The ``1`` of ``max(1, ||T(D, R)||)``: hand-made frames whose two tensors have norms near ``1e-2`` (every
        core scaled down), so the residual is the norm of the difference itself, not a ratio. The dense
        definition again; and the number is far from the ratio, so another floor than 1 would show."""
        for struct in _STRUCTS:
            for C in _STACKS:
                with self.subTest(struct=struct, C=C):
                    frame, _ = library_frame(struct, C, squash_tails=False)
                    d = len(struct[0])
                    hand_made = tuple(tuple(1e-2 ** (0.5 / d) * core / np.sqrt(core[(0,) * len(C)].size)
                                            for core in family) for family in random_like(frame, self.rng))
                    D, _U, L, R = hand_made
                    through_left, through_right = _dense(D, L), _dense(D, R)
                    norm_of_right = _norm_over_modes(through_right, len(C))
                    difference = _norm_over_modes(through_left - through_right, len(C))
                    self.assertLess(float(np.max(norm_of_right)), 0.5)             # below one, in every element
                    residual = fvo.fv_frame_consistency_residual(hand_made)
                    self.assertLess(relative_difference(residual, difference), 1e-10)      # absolute: no division
                    self.assertGreater(float(np.min(difference / norm_of_right / residual)), 2.0)   # not the ratio

    def test_the_tails_are_there_when_asked_for(self):
        """The fixture with open tails really has end bonds above one, in both stores."""
        frame, _ = library_frame(_STRUCTS[1], (), squash_tails=False)
        _, _, L, R = frame
        self.assertEqual((L[0].shape[-3], L[-1].shape[-1]), (2, 3))
        self.assertEqual((R[0].shape[-3], R[-1].shape[-1]), (2, 3))

    def test_a_library_built_frame_passes_at_1e_12(self):
        for struct in _STRUCTS:
            for C in _STACKS:
                for squash_tails in (True, False):
                    with self.subTest(struct=struct, C=C, squash_tails=squash_tails):
                        frame, _ = library_frame(struct, C, squash_tails)
                        self.assertLess(float(np.max(dense_frame_residual(frame, len(C)))), 1e-12)   # it IS consistent
                        self.assertLess(float(np.max(fvo.fv_frame_consistency_residual(frame))), 1e-12)

    def test_the_accuracy_does_not_depend_on_the_norm_of_the_tensor(self):
        """A consistent frame shows a residual at rounding level whether its tensor has norm ``1e-8`` or ``1e14``.
        (Taking the difference across the two gauges as they stand, the residual grows with the norm: ``2e-9`` at
        norm ``4e6``, measured; a difference of inner products stops near ``1e-8`` at every norm.)"""
        struct = ((20, 21, 22, 23, 24), (6, 5, 7, 4, 6), (1, 5, 6, 4, 5, 1))
        for scale in (1e-12, 1e-6, 1.0, 1e3, 1e8):
            with self.subTest(scale=scale):
                frame, _ = library_frame(struct, scale=scale)
                norm = float(np.linalg.norm(frame[3][0]))                 # R is right-orthogonal: its first core
                self.assertTrue(0.1 * scale < norm / 4e6 < 10 * scale, norm)   # the scale reached the tensor
                self.assertLess(float(fvo.fv_frame_consistency_residual(frame)), 1e-12)

    def test_a_relative_perturbation_is_seen_at_its_size_whatever_the_norm(self):
        """The other half of accuracy: a right store off by a relative ``1e-9`` gives a residual near ``1e-9`` at
        norms from 1 up (below 1 the formula's ``max(1, .)`` makes the residual absolute)."""
        struct = ((20, 21, 22, 23, 24), (6, 5, 7, 4, 6), (1, 5, 6, 4, 5, 1))
        for scale in (1e-6, 1.0, 1e8):
            with self.subTest(scale=scale):
                frame, _ = library_frame(struct, scale=scale)
                off = replace_core(frame, 3, 0, (1.0 + 1e-9) * frame[3][0])
                residual = float(fvo.fv_frame_consistency_residual(off))
                norm = float(np.linalg.norm(frame[3][0]))
                expected = 1e-9 * norm / max(1.0, (1.0 + 1e-9) * norm)
                self.assertLess(abs(residual - expected) / expected, 1e-3)

    def test_float32_cores_give_a_float32_residual_at_float32_rounding(self):
        frame, _ = library_frame(_STRUCTS[0], (3,), dtype=np.float32)
        residual = fvo.fv_frame_consistency_residual(frame)
        self.assertEqual(residual.dtype, np.float32)
        self.assertLess(float(np.max(residual)), 1e-5)
        self.assertGreater(float(np.max(residual)), 1e-9)                  # float32 really was the arithmetic
        unstacked, centers = library_frame(_STRUCTS[0], (), dtype=np.float32)       # 0-d: no Python float promotes it
        for residual in (fvo.fv_frame_consistency_residual(unstacked),
                         fvo.fv_centers_consistency_residual(unstacked, centers)):
            self.assertEqual((residual.shape, residual.dtype), ((), np.float32))
            self.assertLess(float(residual), 1e-5)

    def test_a_perturbed_left_or_right_store_fails_at_every_site(self):
        struct = _STRUCTS[0]
        frame, _ = library_frame(struct)
        for family, name in ((2, 'left'), (3, 'right')):
            for site in range(len(struct[0])):
                with self.subTest(store=name, site=site):
                    core = frame[family][site]
                    bad = replace_core(frame, family, site, core * (1.0 + 0.1 * self.rng.standard_normal(core.shape)))
                    residual = float(fvo.fv_frame_consistency_residual(bad))
                    self.assertGreater(residual, 1e-3)
                    self.assertLess(relative_difference(residual, dense_frame_residual(bad, 0)), 1e-10)

    def test_the_up_store_is_not_read(self):
        """``L`` against ``R`` through ``D``: any up cores give the same number (O-5: ``U`` is not checked)."""
        frame, _ = library_frame(_STRUCTS[0], (3,))
        other_up = replace_core(frame, 1, 1, self.rng.standard_normal(frame[1][1].shape))
        self.assertTrue(np.array_equal(fvo.fv_frame_consistency_residual(other_up),
                                       fvo.fv_frame_consistency_residual(frame)))

    def test_it_is_per_stack_element(self):
        """One element of a stack made wrong: that element fails and the others still pass."""
        frame, _ = library_frame(_STRUCTS[0], (2, 2))
        core = frame[2][1].copy()
        core[1, 0] *= 1.0 + 0.1 * self.rng.standard_normal(core.shape[2:])
        residual = fvo.fv_frame_consistency_residual(replace_core(frame, 2, 1, core))
        self.assertEqual(residual.shape, (2, 2))
        wrong = np.zeros((2, 2), dtype=bool)
        wrong[1, 0] = True
        self.assertTrue(np.array_equal(residual > 1e-3, wrong))
        self.assertLess(float(np.max(residual[~wrong])), 1e-12)

    def test_a_zero_frame_has_residual_zero(self):
        frame, _ = library_frame(_STRUCTS[0])
        zero = tuple(tuple(np.zeros_like(core) for core in family) for family in frame)
        self.assertEqual(float(fvo.fv_frame_consistency_residual(zero)), 0.0)


class TestRaggedCentersResidual(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.rng = np.random.default_rng(1)

    def test_it_is_the_definition_on_hand_made_frames_and_centers(self):
        """Random frame, random centers; and the library's frame with random centers; each family alone. The
        residual is the largest of the dense mismatches over the sites and the families given."""
        for struct in _STRUCTS:
            for C in _STACKS:
                for squash_tails in (True, False):
                    frame, centers = library_frame(struct, C, squash_tails)
                    hand_frame, hand_centers = random_like(frame, self.rng), random_like(centers, self.rng)
                    for name, args in (('both', (hand_frame, hand_centers)),
                                       ('Vc alone', (hand_frame, (hand_centers[0], None))),
                                       ('Hc alone', (hand_frame, (None, hand_centers[1]))),
                                       ('library frame', (frame, hand_centers))):
                        with self.subTest(struct=struct, C=C, squash_tails=squash_tails, case=name):
                            residual = fvo.fv_centers_consistency_residual(*args)
                            self.assertEqual(residual.shape, C)
                            expected = dense_centers_residual(*args, len(C))
                            self.assertGreater(float(np.min(expected)), 1e-3)
                            self.assertLess(relative_difference(residual, expected), 1e-12)

    def test_the_library_s_centers_pass_at_1e_12(self):
        """The second result of ``t3_orthogonal_representations`` is the base point written into the frame's hole,
        at every site and in both families: by the dense definition, and by the checker."""
        for struct in _STRUCTS:
            for C in _STACKS:
                for squash_tails in (True, False):
                    with self.subTest(struct=struct, C=C, squash_tails=squash_tails):
                        frame, centers = library_frame(struct, C, squash_tails)
                        self.assertLess(float(np.max(dense_centers_residual(frame, centers, len(C)))), 1e-12)
                        self.assertLess(float(np.max(fvo.fv_centers_consistency_residual(frame, centers))), 1e-12)

    def test_the_accuracy_does_not_depend_on_the_norm_of_the_tensor(self):
        struct = ((20, 21, 22, 23, 24), (6, 5, 7, 4, 6), (1, 5, 6, 4, 5, 1))
        for scale in (1e-12, 1e-6, 1.0, 1e3, 1e8):
            with self.subTest(scale=scale):
                frame, centers = library_frame(struct, scale=scale)
                self.assertLess(float(fvo.fv_centers_consistency_residual(frame, centers)), 1e-12)

    def test_the_accuracy_does_not_depend_on_the_scale_of_a_down_core(self):
        """``D[i]`` scaled, and ``L[i]``, ``R[i]``, ``Hc[i]`` over it scaled back: every tensor of the definition
        is the same, so consistent centers still read rounding level. The library builds ``D`` at unit scale;
        a hand-made frame need not. Pins the balancing of ``D[i]`` against ``Vc[i]`` where the two are factored
        together (without it this reads about ``1e-8`` at a scale of ``1e8``)."""
        struct = ((20, 21, 22, 23, 24), (6, 5, 7, 4, 6), (1, 5, 6, 4, 5, 1))
        for scale in (1e-8, 1e8):
            for site in range(len(struct[0])):
                with self.subTest(scale=scale, site=site):
                    (DD, UU, LL, RR), (Vc, Hc) = library_frame(struct)
                    at_site = lambda cores, factor: tuple(factor * core if ii == site else core
                                                          for ii, core in enumerate(cores))
                    frame = (at_site(DD, scale), UU, at_site(LL, 1.0 / scale), at_site(RR, 1.0 / scale))
                    centers = (Vc, at_site(Hc, 1.0 / scale))
                    self.assertLess(float(fvo.fv_centers_consistency_residual(frame, centers)), 1e-12)
                    self.assertLess(float(fvo.fv_frame_consistency_residual(frame)), 1e-12)

    def test_float32_gives_a_float32_residual_at_float32_rounding(self):
        frame, centers = library_frame(_STRUCTS[0], (3,), dtype=np.float32)
        residual = fvo.fv_centers_consistency_residual(frame, centers)
        self.assertEqual(residual.dtype, np.float32)
        self.assertLess(float(np.max(residual)), 1e-5)

    def test_a_wrong_center_fails_at_every_site_in_either_family(self):
        """One center of one site replaced by that of another point: caught with both families given and with
        the wrong one's family alone; the other family alone still passes."""
        struct = _STRUCTS[0]
        frame, centers = library_frame(struct)
        _, other_centers = library_frame(struct)                     # the centers of another point, same shapes
        for family, name in ((0, 'Vc'), (1, 'Hc')):
            for site in range(len(struct[0])):
                with self.subTest(family=name, site=site):
                    wrong = replace_core(centers, family, site, other_centers[family][site])
                    alone = tuple(wrong[ff] if ff == family else None for ff in (0, 1))
                    other_alone = tuple(None if ff == family else wrong[ff] for ff in (0, 1))
                    residual = float(fvo.fv_centers_consistency_residual(frame, wrong))
                    self.assertGreater(residual, 1e-2)
                    self.assertLess(relative_difference(residual, dense_centers_residual(frame, wrong, 0)), 1e-10)
                    self.assertGreater(float(fvo.fv_centers_consistency_residual(frame, alone)), 1e-2)
                    self.assertLess(float(fvo.fv_centers_consistency_residual(frame, other_alone)), 1e-12)

    def test_a_slightly_wrong_center_is_seen_at_its_size(self):
        frame, centers = library_frame(_STRUCTS[0])
        for family in (0, 1):
            wrong = replace_core(centers, family, 2, (1.0 + 1e-8) * centers[family][2])
            residual = float(fvo.fv_centers_consistency_residual(frame, wrong))
            self.assertLess(abs(residual - 1e-8) / 1e-8, 1e-3)       # the whole tensor is off by a factor 1 + 1e-8

    def test_the_centers_of_a_frame_whose_right_or_up_store_is_wrong_fail(self):
        """The centers are read through ``R`` to the right of their site and, for ``Vc``, through ``U``: with the
        library's centers and a wrong ``R`` or ``U`` the networks are not the base point."""
        frame, centers = library_frame(_STRUCTS[0])
        wrong_R = replace_core(frame, 3, 2, frame[3][2] * (1.0 + 0.1 * self.rng.standard_normal(frame[3][2].shape)))
        wrong_U = replace_core(frame, 1, 1, frame[1][1] * (1.0 + 0.1 * self.rng.standard_normal(frame[1][1].shape)))
        self.assertGreater(float(fvo.fv_centers_consistency_residual(wrong_R, centers)), 1e-3)
        self.assertGreater(float(fvo.fv_centers_consistency_residual(wrong_U, centers)), 1e-3)
        self.assertLess(float(fvo.fv_centers_consistency_residual(wrong_U, (None, centers[1]))), 1e-12)   # Hc: no U

    def test_it_is_per_stack_element(self):
        frame, centers = library_frame(_STRUCTS[0], (2, 2))
        core = centers[0][1].copy()
        core[0, 1] *= 1.5
        residual = fvo.fv_centers_consistency_residual(frame, replace_core(centers, 0, 1, core))
        wrong = np.zeros((2, 2), dtype=bool)
        wrong[0, 1] = True
        self.assertTrue(np.array_equal(residual > 1e-2, wrong))
        self.assertLess(float(np.max(residual[~wrong])), 1e-12)

    def test_no_centers_at_all_is_a_structural_error(self):
        frame, _ = library_frame(_STRUCTS[0])
        with self.assertRaises(ValueError):
            fvo.fv_centers_consistency_residual(frame, (None, None))


class TestNormsThatDoNotOverflow(unittest.TestCase):
    """A float32 frame of norm ``1e20``: the square of its norm (``1e40``) is not a float32. A norm taken by a
    plain sum of squares is ``inf`` there, and a residual ``num / inf`` reads 0: a pass for any frame."""
    _STRUCT = ((20, 21, 22, 23, 24), (6, 5, 7, 4, 6), (1, 5, 6, 4, 5, 1))      # norm 4e6 at scale 1

    def setUp(self):
        np.random.seed(5)

    def _frames(self, scale, C=()):
        frame, centers = library_frame(self._STRUCT, C, scale=scale, dtype=np.float32)
        norm = np.linalg.norm(np.asarray(frame[3][0], dtype=np.float64).reshape(C + (-1,)), axis=-1)
        return frame, centers, norm

    def test_a_consistent_float32_frame_of_norm_1e20_passes(self):
        for scale in (2.5e13, 2.5e30):                                  # norms near 1e20 and 1e37
            with self.subTest(scale=scale):
                frame, centers, norm = self._frames(scale)
                self.assertGreater(float(norm), 1.9e19)                 # the square overflows float32
                for residual in (fvo.fv_frame_consistency_residual(frame),
                                 fvo.fv_centers_consistency_residual(frame, centers)):
                    self.assertEqual(residual.dtype, np.float32)
                    self.assertLess(float(residual), 1e-5)
                    self.assertGreater(float(residual), 0.0)            # a rounding-level number, not a 0 from 1/inf

    def test_an_inconsistent_float32_frame_of_norm_1e20_reads_its_size(self):
        """``R[0]`` off by a relative ``1e-3``; ``Vc[1]`` off by ``1e-3``; ``Hc[2]`` off by ``1e-3``."""
        for scale in (2.5e13, 2.5e30):
            with self.subTest(scale=scale):
                frame, centers, _ = self._frames(scale)
                off = np.float32(1.001)
                bad_frame = replace_core(frame, 3, 0, off * frame[3][0])
                for name, residual in (
                        ('frame', fvo.fv_frame_consistency_residual(bad_frame)),
                        ('Vc', fvo.fv_centers_consistency_residual(frame, replace_core(centers, 0, 1, off * centers[0][1]))),
                        ('Hc', fvo.fv_centers_consistency_residual(frame, replace_core(centers, 1, 2, off * centers[1][2])))):
                    self.assertEqual(residual.dtype, np.float32, name)
                    self.assertLess(abs(float(residual) - 1e-3), 2e-5, name)

    def test_a_norm_that_is_not_representable_gives_nan_for_its_stack_element_only(self):
        """A stack of three: the middle element's right store scaled so that ``||T(D, R)||`` exceeds the largest
        float32 (its cores stay finite). That element is ``nan``, which fails every threshold; the others keep
        their residuals."""
        frame, centers, norm = self._frames(2.5e13, (3,))
        good = fvo.fv_frame_consistency_residual(frame)
        first = frame[3][0].copy()
        second = frame[3][1].copy()
        first[1] *= np.float32(1e15)
        second[1] *= np.float32(1e15)                                   # 1e20 * 1e30: beyond 3.4e38, cores finite
        too_large = replace_core(replace_core(frame, 3, 0, first), 3, 1, second)
        self.assertTrue(all(np.all(np.isfinite(core)) for family in too_large for core in family))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)             # numpy's own overflow warnings
            residual = fvo.fv_frame_consistency_residual(too_large)
        self.assertEqual((residual.shape, residual.dtype), ((3,), np.float32))
        self.assertTrue(np.isnan(residual[1]))
        self.assertFalse(residual[1] <= 1e30)
        self.assertLess(float(np.max(np.abs(residual[[0, 2]] - good[[0, 2]]))), 1e-6)   # (recomputed alone if the SVD raised)
        self.assertLess(float(np.max(residual[[0, 2]])), 1e-5)

    def test_the_uniform_twins_do_not_overflow_either(self):
        frame, centers, norm = self._frames(2.5e13)
        (D, U, L, R), (Vc, Hc) = frame, centers
        d, shape = len(D), self._STRUCT[0]
        pad = lambda cores, widths: np.stack([np.pad(c, [(0, w - s) for s, w in zip(c.shape, widths)]) for c in cores])
        nD, nU = max(c.shape[0] for c in D), max(c.shape[1] for c in U)
        rL, rR, N = max(max(c.shape[0], c.shape[2]) for c in L), max(max(c.shape[0], c.shape[2]) for c in R), max(shape)
        masks = (_prefix_mask([c.shape[0] for c in D], nD), _prefix_mask([c.shape[1] for c in U], nU),
                 _prefix_mask([c.shape[0] for c in L] + [L[-1].shape[2]], rL),
                 _prefix_mask([c.shape[0] for c in R] + [R[-1].shape[2]], rR))
        data = (pad(D, (nD, N)), pad(U, (rL, nU, rR)), pad(L, (rL, nD, rL)), pad(R, (rR, nD, rR)), shape, masks)
        uniform_centers = (pad(Vc, (nU, N)), pad(Hc, (rL, nD, rR)))
        self.assertEqual(data[0].dtype, np.float32)
        self.assertLess(float(ufvo.ufv_frame_consistency_residual(data)), 1e-5)
        self.assertLess(float(ufvo.ufv_centers_consistency_residual(data, uniform_centers)), 1e-5)
        bad_right = data[3].copy()
        bad_right[0] *= np.float32(1.001)
        residual = ufvo.ufv_frame_consistency_residual(data[:3] + (bad_right,) + data[4:])
        self.assertEqual(residual.dtype, np.float32)
        self.assertLess(abs(float(residual) - 1e-3), 2e-5)


class TestNoDenseTensor(unittest.TestCase):
    """``d = 12``, ``N = 100``: the dense tensor has ``1e24`` entries and could not be allocated. Each checker
    returns, with the right number."""

    def setUp(self):
        np.random.seed(0)
        struct = ((100,) * 12, (5,) * 12, (1,) + (4,) * 11 + (1,))
        self.x = t3_constructors.t3_corewise_randn(*struct)
        self.frame, self.centers = fv_conversions.t3_orthogonal_representations(self.x)

    def test_the_ragged_checkers(self):
        self.assertLess(float(fvo.fv_frame_consistency_residual(self.frame)), 1e-12)
        self.assertLess(float(fvo.fv_centers_consistency_residual(self.frame, self.centers)), 1e-12)
        bad = replace_core(self.frame, 3, 5, 1.01 * self.frame[3][5])
        self.assertLess(abs(float(fvo.fv_frame_consistency_residual(bad)) - 0.01 / 1.01), 1e-10)

    def test_the_uniform_checkers(self):
        tucker_sc, tt_sc = np.stack(self.x[0]), np.stack([np.pad(G, [(0, 4 - G.shape[0]), (0, 0), (0, 4 - G.shape[2])])
                                                           for G in self.x[1]])
        masks = ut3_masking.ut3_make_masks(np.full(12, 5), np.array([1] + [4] * 11 + [1]), 5, 4)
        frame_data, (Vc_sc, Hc_sc, _, _) = ufv_conversions.ut3_orthogonal_representations(
            (tucker_sc, tt_sc, (100,) * 12, masks))
        self.assertLess(float(ufvo.ufv_frame_consistency_residual(frame_data)), 1e-12)
        self.assertLess(float(ufvo.ufv_centers_consistency_residual(frame_data, (Vc_sc, Hc_sc))), 1e-12)


# ---------------------------------------------------------------------------------- the uniform twins
def _prefix_mask(ranks, pad):  # ranks: HOST int (...,) -> bool (..., pad)
    return np.arange(pad) < np.asarray(ranks)[..., None]


def hand_made_uniform(rng, C, tails):
    """A hand-made uniform frame and centers: random supercores (the padding included, so it is garbage), prefix
    masks whose ranks are drawn independently for every stack element, mode sizes below the padded ``N``."""
    d, N, nD, nU, rL, rR = 3, 7, 4, 5, 4, 3
    shape = (4, 7, 5)
    draw = lambda low, high, length: rng.integers(low, high + 1, size=(length,) + C)
    down_r, up_r = draw(1, nD, d), draw(1, nU, d)
    left_r, right_r = draw(1, rL, d + 1), draw(1, rR, d + 1)
    if not tails:
        left_r[0] = left_r[-1] = right_r[0] = right_r[-1] = 1
    masks = (_prefix_mask(down_r, nD), _prefix_mask(up_r, nU), _prefix_mask(left_r, rL), _prefix_mask(right_r, rR))
    supercores = (rng.standard_normal((d,) + C + (nD, N)), rng.standard_normal((d,) + C + (rL, nU, rR)),
                  rng.standard_normal((d,) + C + (rL, nD, rL)), rng.standard_normal((d,) + C + (rR, nD, rR)))
    centers = (rng.standard_normal((d,) + C + (nU, N)), rng.standard_normal((d,) + C + (rL, nD, rR)))
    return supercores + (shape, masks), centers, (down_r, up_r, left_r, right_r)


def ragged_element(data, centers, ranks, c):
    """Stack element ``c`` of a uniform frame and centers as ragged cores, sliced to the REAL ranks and mode sizes
    by this file's own indexing (not by the library's converter, and not by masking)."""
    down_sc, up_sc, left_sc, right_sc, shape, _ = data
    Vc_sc, Hc_sc = centers
    down_r, up_r, left_r, right_r = ranks
    at = lambda sc, ii: sc[(ii,) + c]
    r = lambda ranks_, ii: int(ranks_[(ii,) + c])
    d = len(shape)
    frame = (tuple(at(down_sc, ii)[:r(down_r, ii), :shape[ii]] for ii in range(d)),
             tuple(at(up_sc, ii)[:r(left_r, ii), :r(up_r, ii), :r(right_r, ii + 1)] for ii in range(d)),
             tuple(at(left_sc, ii)[:r(left_r, ii), :r(down_r, ii), :r(left_r, ii + 1)] for ii in range(d)),
             tuple(at(right_sc, ii)[:r(right_r, ii), :r(down_r, ii), :r(right_r, ii + 1)] for ii in range(d)))
    ragged_centers = (tuple(at(Vc_sc, ii)[:r(up_r, ii), :shape[ii]] for ii in range(d)),
                      tuple(at(Hc_sc, ii)[:r(left_r, ii), :r(down_r, ii), :r(right_r, ii + 1)] for ii in range(d)))
    return frame, ragged_centers


def add_garbage(data, centers, rng, scale=1e3):
    """Garbage added to every padded slot of the four frame supercores and the two centers' supercores; the real
    slots keep their bits. The indicator of the real slots is built here from the masks, by outer products."""
    down_sc, up_sc, left_sc, right_sc, shape, (dm, um, lm, rm) = data
    Vc_sc, Hc_sc = centers
    d, N = down_sc.shape[0], down_sc.shape[-1]
    C = down_sc.shape[1:-2]
    sm = _prefix_mask(np.asarray(shape), N).reshape((d,) + (1,) * len(C) + (N,))
    outer2 = lambda a, b: a[..., :, None] & b[..., None, :]
    outer3 = lambda a, b, c: a[..., :, None, None] & b[..., None, :, None] & c[..., None, None, :]
    real = (outer2(dm, sm), outer3(lm[:-1], um, rm[1:]), outer3(lm[:-1], dm, lm[1:]), outer3(rm[:-1], dm, rm[1:]),
            outer2(um, sm), outer3(lm[:-1], dm, rm[1:]))
    dirty = [np.where(keep, sc, sc + scale * rng.standard_normal(sc.shape))
             for sc, keep in zip((down_sc, up_sc, left_sc, right_sc, Vc_sc, Hc_sc), real)]
    self_check = all(np.array_equal(a[keep], b[keep]) and not np.array_equal(a, b)
                     for a, b, keep in zip(dirty, (down_sc, up_sc, left_sc, right_sc, Vc_sc, Hc_sc), real))
    assert self_check, 'the garbage fixture must change the padding and only the padding'
    return tuple(dirty[:4]) + (shape, (dm, um, lm, rm)), tuple(dirty[4:])


class TestUniformTwins(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)
        self.rng = np.random.default_rng(2)

    def test_they_equal_the_ragged_functions_element_by_element(self):
        """Hand-made uniform frames and centers, ranks varying over the stack, garbage in the padding, with and
        without tails: each uniform residual is the ragged function's on that element's real cores (and, through
        the tests above, the dense definition)."""
        for C in [(), (3,), (2, 2)]:
            for tails in (False, True):
                with self.subTest(C=C, tails=tails):
                    data, centers, ranks = hand_made_uniform(self.rng, C, tails)
                    frame_residual = ufvo.ufv_frame_consistency_residual(data)
                    centers_residual = ufvo.ufv_centers_consistency_residual(data, centers)
                    Vc_residual = ufvo.ufv_centers_consistency_residual(data, (centers[0], None))
                    Hc_residual = ufvo.ufv_centers_consistency_residual(data, (None, centers[1]))
                    for residual in (frame_residual, centers_residual, Vc_residual, Hc_residual):
                        self.assertEqual(residual.shape, C)
                    for c in np.ndindex(*C):
                        frame, ragged_centers = ragged_element(data, centers, ranks, c)
                        Vc, Hc = ragged_centers
                        for got, expected in (
                                (frame_residual[c], fvo.fv_frame_consistency_residual(frame)),
                                (frame_residual[c], dense_frame_residual(frame, 0)),
                                (centers_residual[c], fvo.fv_centers_consistency_residual(frame, ragged_centers)),
                                (centers_residual[c], dense_centers_residual(frame, ragged_centers, 0)),
                                (Vc_residual[c], fvo.fv_centers_consistency_residual(frame, (Vc, None))),
                                (Hc_residual[c], fvo.fv_centers_consistency_residual(frame, (None, Hc)))):
                            self.assertGreater(float(expected), 1e-3)
                            self.assertLess(relative_difference(got, expected), 1e-11)

    def test_garbage_in_the_padding_changes_nothing(self):
        """The frame is masked with its own masks and the centers with its variation masks, once: garbage of size
        ``1e3`` in every padded slot leaves each residual bit for bit."""
        for C in [(), (3,), (2, 2)]:
            for tails in (False, True):
                with self.subTest(C=C, tails=tails):
                    data, centers, _ = hand_made_uniform(self.rng, C, tails)
                    dirty_data, dirty_centers = add_garbage(data, centers, self.rng)
                    self.assertTrue(np.array_equal(ufvo.ufv_frame_consistency_residual(dirty_data),
                                                   ufvo.ufv_frame_consistency_residual(data)))
                    for pick in ((0, 1), (0, None), (None, 1)):
                        choose = lambda both: tuple(None if ii is None else both[ii] for ii in pick)
                        self.assertTrue(np.array_equal(
                            ufvo.ufv_centers_consistency_residual(dirty_data, choose(dirty_centers)),
                            ufvo.ufv_centers_consistency_residual(data, choose(centers))))

    def test_a_nan_or_an_inf_in_the_padding_changes_nothing(self):
        """The padding is don't-care whatever it holds: with every padded slot of the frame and of the centers set
        to ``nan``, or to ``inf``, each residual is the clean one bit for bit (a mask applied by multiplication
        would give ``nan * 0 = nan``)."""
        for C in [(), (3,)]:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(C=C, value=value):
                    data, centers, _ = hand_made_uniform(self.rng, C, True)
                    dirty_data, dirty_centers = add_garbage(data, centers, self.rng)
                    padding = [a != b for a, b in zip(dirty_data[:4] + dirty_centers, data[:4] + centers)]
                    poisoned = [np.where(pad, value, sc) for sc, pad in zip(data[:4] + centers, padding)]
                    self.assertTrue(all(np.any(~np.isfinite(sc)) for sc in poisoned))
                    bad_data, bad_centers = tuple(poisoned[:4]) + data[4:], tuple(poisoned[4:])
                    self.assertTrue(np.array_equal(ufvo.ufv_frame_consistency_residual(bad_data),
                                                   ufvo.ufv_frame_consistency_residual(data)))
                    for pick in ((0, 1), (0, None), (None, 1)):
                        choose = lambda both: tuple(None if ii is None else both[ii] for ii in pick)
                        self.assertTrue(np.array_equal(
                            ufvo.ufv_centers_consistency_residual(bad_data, choose(bad_centers)),
                            ufvo.ufv_centers_consistency_residual(data, choose(centers))))

    def test_a_library_built_uniform_frame_and_its_centers_pass(self):
        """Over the stack matrix: no stack, a stack, a two-axis stack, forced-larger padding, ranks that vary over
        the stack. The uniform frame is gauge-equivalent to the ragged one, not identical, so the residuals are
        compared with the tolerance, not the frames with each other."""
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        struct = ((5, 7, 6), (2, 3, 2), (1, 2, 2, 1))
        low = ((5, 7, 6), (1, 2, 2), (1, 1, 2, 1))
        cases = {
            'unstacked': ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*struct)),
            'stack': ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*struct, stack_shape=(2,))),
            'two stack axes': ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*struct, stack_shape=(2, 3))),
            'forced padding': ut3.UniformTuckerTensorTrain.from_t3(
                t3.TuckerTensorTrain.randn(*struct, stack_shape=(2,)), N=9, n=5, r=5),
            'ranks varying over the stack': ut3.UniformTuckerTensorTrain.stack([
                ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*struct), n=3, r=2),
                ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*low), n=3, r=2)]),
        }
        for name, ux in cases.items():
            with self.subTest(case=name):
                frame_data, (Vc_sc, Hc_sc, _, _) = ufv_conversions.ut3_orthogonal_representations(ux.data)
                self.assertEqual(ufvo.ufv_frame_consistency_residual(frame_data).shape, ux.stack_shape)
                self.assertLess(float(np.max(ufvo.ufv_frame_consistency_residual(frame_data))), 1e-12)
                self.assertLess(float(np.max(ufvo.ufv_centers_consistency_residual(frame_data, (Vc_sc, Hc_sc)))), 1e-12)
                dirty_data, (dirty_Vc, dirty_Hc) = add_garbage(frame_data, (Vc_sc, Hc_sc), self.rng)
                self.assertLess(float(np.max(ufvo.ufv_frame_consistency_residual(dirty_data))), 1e-12)
                self.assertLess(float(np.max(ufvo.ufv_centers_consistency_residual(dirty_data, (dirty_Vc, dirty_Hc)))), 1e-12)
        self.assertFalse(np.array_equal(cases['ranks varying over the stack'].masks.tt_edge_mask[1, 0],
                                        cases['ranks varying over the stack'].masks.tt_edge_mask[1, 1]))   # they do vary

    def test_a_wrong_real_slot_fails_only_its_stack_element(self):
        import t3toolbox.tucker_tensor_train as t3
        import t3toolbox.uniform_tucker_tensor_train as ut3
        ux = ut3.UniformTuckerTensorTrain.from_t3(
            t3.TuckerTensorTrain.randn((5, 7, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=(3,)), N=9, n=5, r=5)
        frame_data, (Vc_sc, Hc_sc, _, _) = ufv_conversions.ut3_orthogonal_representations(ux.data)
        down_sc, up_sc, left_sc, right_sc, shape, masks = frame_data
        wrong = np.array([False, True, False])
        bad_right = right_sc.copy()
        bad_right[1, 1, 0, 0, 0] += 0.5                                  # site 1, stack element 1, a real slot
        self.assertTrue(masks[3][1, 1, 0] and masks[0][1, 1, 0] and masks[3][2, 1, 0])
        residual = ufvo.ufv_frame_consistency_residual((down_sc, up_sc, left_sc, bad_right, shape, masks))
        self.assertTrue(np.array_equal(residual > 1e-3, wrong))
        self.assertLess(float(np.max(residual[~wrong])), 1e-12)
        bad_Vc, bad_Hc = Vc_sc.copy(), Hc_sc.copy()
        bad_Vc[2, 1, 0, 0] += 0.5
        bad_Hc[0, 1, 0, 0, 0] += 0.5
        for bad_centers in ((bad_Vc, Hc_sc), (Vc_sc, bad_Hc), (bad_Vc, None), (None, bad_Hc)):
            residual = ufvo.ufv_centers_consistency_residual(frame_data, bad_centers)
            self.assertTrue(np.array_equal(residual > 1e-3, wrong))
            self.assertLess(float(np.max(residual[~wrong])), 1e-12)

    def test_no_centers_at_all_is_a_structural_error(self):
        data, _, _ = hand_made_uniform(self.rng, (), False)
        with self.assertRaises(ValueError):
            ufvo.ufv_centers_consistency_residual(data, (None, None))


class TestCheckersNotGuarantees(unittest.TestCase):
    """The backend gives checkers, not guarantees: a checker returns its residual whatever the numbers, and no
    other backend function calls one."""

    def test_a_checker_returns_a_residual_on_any_numbers(self):
        rng = np.random.default_rng(3)
        frame, centers = library_frame(_STRUCTS[0], (2,))
        wild = random_like(frame, rng)
        self.assertTrue(np.all(fvo.fv_frame_consistency_residual(wild) > 1e-3))                  # no error: a number
        self.assertTrue(np.all(fvo.fv_centers_consistency_residual(wild, random_like(centers, rng)) > 1e-3))
        huge = replace_core(frame, 3, 0, 1e100 * frame[3][0])
        self.assertTrue(np.allclose(fvo.fv_frame_consistency_residual(huge), 1.0))               # ||L - R|| / ||R||

    def test_the_guard_zeroes_the_stack_elements_that_are_not_finite(self):
        """The guard by itself, with no SVD in reach: the flag is per stack element, over every core of every
        family; an element with a ``nan`` or an ``inf`` anywhere comes back as zeros, the others bit for bit."""
        rng = np.random.default_rng(4)
        frame, _ = library_frame(_STRUCTS[0], (2, 2))
        for value in (np.nan, np.inf, -np.inf):
            for family, site in ((0, 0), (1, 1), (2, 2), (3, 1)):
                with self.subTest(value=value, family=family, site=site):
                    core = frame[family][site].copy()
                    core[(1, 0) + (-1,) * (core.ndim - 2)] = value                  # stack element (1, 0), last entry
                    bad = replace_core(frame, family, site, core)
                    finite_C, cleaned = fvo._zero_stack_elements_that_are_not_finite(bad, 2)
                    expected = np.ones((2, 2), dtype=bool)
                    expected[1, 0] = False
                    self.assertTrue(np.array_equal(finite_C, expected))
                    for cleaned_family, family_cores in zip(cleaned, frame):
                        for cleaned_core, good_core in zip(cleaned_family, family_cores):
                            self.assertTrue(np.all(np.isfinite(cleaned_core)))
                            self.assertTrue(np.all(cleaned_core[1, 0] == 0.0))
                            self.assertTrue(np.array_equal(cleaned_core[expected], good_core[expected]))
        finite_C, cleaned = fvo._zero_stack_elements_that_are_not_finite(frame, 2)
        self.assertTrue(finite_C.all())
        self.assertTrue(all(np.array_equal(a, b) for fa, fb in zip(cleaned, frame) for a, b in zip(fa, fb)))

    def test_cores_that_are_not_finite_give_nan_and_no_error(self):
        """numpy's SVD raises on a matrix with a ``nan`` and can run forever on one with an ``inf``. The checkers
        return: the stack element with the bad core gets ``nan`` (which fails every threshold) and the other
        elements keep their residual. So that a broken guard FAILS this test instead of hanging it, the
        orthogonalization is wrapped to refuse anything that is not finite before an SVD sees it."""
        rng = np.random.default_rng(4)
        frame, centers = library_frame(_STRUCTS[0], (2,))
        good_frame = fvo.fv_frame_consistency_residual(frame)
        good_centers = fvo.fv_centers_consistency_residual(frame, centers)
        data, uniform_centers, _ = hand_made_uniform(rng, (2,), True)
        good_uniform = (ufvo.ufv_frame_consistency_residual(data), ufvo.ufv_centers_consistency_residual(data, uniform_centers))
        real_orthogonalize, real_factors = fvo._tt_left_orthogonal_form, fvo._tucker_down_factors

        def finite_only(real):
            def wrapped(cores):
                if not all(np.all(np.isfinite(core)) for core in cores):
                    raise AssertionError('a core that is not finite reached the orthogonalization')
                return real(cores)
            return wrapped

        with mock.patch.object(fvo, '_tt_left_orthogonal_form', finite_only(real_orthogonalize)), \
                mock.patch.object(fvo, '_tucker_down_factors', finite_only(real_factors)):
            for value in (np.nan, np.inf, -np.inf):
                for family in (0, 2, 3):                           # D, L, R: the stores the frame's residual reads
                    with self.subTest(value=value, family=family):
                        core = frame[family][1].copy()
                        core[(1,) + (0,) * (core.ndim - 1)] = value            # stack element 1
                        bad = replace_core(frame, family, 1, core)
                        for residual, good in ((fvo.fv_frame_consistency_residual(bad), good_frame),
                                               (fvo.fv_centers_consistency_residual(bad, centers), good_centers)):
                            self.assertEqual((residual.shape, residual.dtype), ((2,), np.float64))
                            self.assertTrue(np.isnan(residual[1]) and not residual[1] <= 1e300)
                            self.assertEqual(residual[0], good[0])
            bad_up = frame[1][1].copy()
            bad_up[1, 0, 0, 0] = np.inf
            bad = replace_core(frame, 1, 1, bad_up)
            self.assertTrue(np.array_equal(fvo.fv_frame_consistency_residual(bad), good_frame))       # U is not read
            self.assertTrue(np.array_equal(fvo.fv_centers_consistency_residual(bad, (None, centers[1])),
                                           fvo.fv_centers_consistency_residual(frame, (None, centers[1]))))
            self.assertTrue(np.isnan(fvo.fv_centers_consistency_residual(bad, centers)[1]))
            for family in (0, 1):
                bad_center = centers[family][2].copy()
                bad_center[(0,) * bad_center.ndim] = np.inf                    # stack element 0
                residual = fvo.fv_centers_consistency_residual(frame, replace_core(centers, family, 2, bad_center))
                self.assertTrue(np.isnan(residual[0]))
                self.assertEqual(residual[1], good_centers[1])
            bad_left = data[2].copy()
            bad_left[1, 0, 0, 0, 0] = np.inf                       # stack element 0, a real slot (every rank is >= 1)
            bad_data = data[:2] + (bad_left,) + data[3:]
            for residual, good in zip((ufvo.ufv_frame_consistency_residual(bad_data),
                                       ufvo.ufv_centers_consistency_residual(bad_data, uniform_centers)), good_uniform):
                self.assertTrue(np.isnan(residual[0]))
                self.assertEqual(residual[1], good[1])

    def test_finite_cores_whose_products_overflow_give_nan_for_their_stack_element_only(self):
        """Every core of stack element 1 scaled by ``1e200``: the products overflow inside the sweep and numpy's
        SVD raises. The checkers return ``nan`` for that element and the residual of the other (as jax does by
        itself); unstacked, ``nan``."""
        frame, centers = library_frame(_STRUCTS[0], (2,))
        scale = np.array([1.0, 1e200])
        huge = tuple(tuple(core * scale.reshape((2,) + (1,) * (core.ndim - 1)) for core in family) for family in frame)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)                # numpy's own overflow warnings
            for residual, good in ((fvo.fv_frame_consistency_residual(huge), fvo.fv_frame_consistency_residual(frame)),
                                   (fvo.fv_centers_consistency_residual(huge, centers),
                                    fvo.fv_centers_consistency_residual(frame, centers))):
                self.assertEqual((residual.shape, residual.dtype), ((2,), np.float64))
                self.assertTrue(np.isnan(residual[1]))
                self.assertLess(abs(float(residual[0]) - float(good[0])), 1e-13)   # recomputed alone: rounding apart
            unstacked = tuple(tuple(core[1] for core in family) for family in huge)
            residual = fvo.fv_frame_consistency_residual(unstacked)
            self.assertEqual((residual.shape, residual.dtype), ((), np.float64))
            self.assertTrue(np.isnan(residual))

    def test_the_failure_path_takes_one_stack_element_at_a_time(self):
        """The failure path by itself: a computation that raises ``LinAlgError`` whenever stack element ``(1, 0)``
        is among its input gets ``nan`` there and its values elsewhere; any other error is not swallowed."""
        cores = ((np.arange(8.0).reshape(2, 2, 2, 1),),)                   # C = (2, 2), one family, one core

        def residual_of(families):
            core = families[0][0]
            if np.any(core == 4.0):                                        # the first entry of element (1, 0)
                raise np.linalg.LinAlgError('SVD did not converge')
            return core.sum(axis=(-2, -1))

        out = fvo._per_stack_element_if_the_svd_fails(residual_of, cores, 2)
        self.assertTrue(np.isnan(out[1, 0]))
        self.assertTrue(np.array_equal(out[[0, 0, 1], [0, 1, 1]], np.array([1.0, 5.0, 13.0])))
        def other_error_on_the_stack(families):                            # only LinAlgError is the guard's to take
            if families[0][0].ndim == 4:
                raise ZeroDivisionError
            return families[0][0].sum(axis=(-2, -1))

        def other_error_on_an_element(families):
            if families[0][0].ndim == 4:
                raise np.linalg.LinAlgError('SVD did not converge')
            raise ZeroDivisionError

        for residual_of in (other_error_on_the_stack, other_error_on_an_element):
            with self.assertRaises(ZeroDivisionError):
                fvo._per_stack_element_if_the_svd_fails(residual_of, cores, 2)

    def test_a_norm_that_is_not_finite_gives_nan_and_never_zero(self):
        """``num / max(1, den)`` by itself: an ``inf`` denominator under a finite numerator would read 0 (a pass);
        it reads ``nan``, as does an ``inf`` or ``nan`` numerator. The dtype is the norms'."""
        num = np.array([3.0, 3.0, np.inf, np.nan, 3.0, 0.25], dtype=np.float32)
        den = np.array([6.0, np.inf, 6.0, 6.0, np.nan, 0.5], dtype=np.float32)
        residual = fvo._relative_residual(num, den)
        self.assertEqual(residual.dtype, np.float32)
        self.assertTrue(np.array_equal(np.isnan(residual), np.array([False, True, True, True, True, False])))
        self.assertTrue(np.array_equal(residual[[0, 5]], np.array([0.5, 0.25], dtype=np.float32)))
        unstacked = fvo._relative_residual(np.float32(3.0) * np.ones((), np.float32), np.ones((), np.float32))
        self.assertEqual((unstacked.shape, unstacked.dtype), ((), np.float32))

    def test_no_other_backend_function_calls_a_checker(self):
        """A search of the backend's source for the four names: the only uses outside the four definitions are the
        two uniform twins calling their ragged functions."""
        names = {'fv_frame_consistency_residual', 'fv_centers_consistency_residual',
                 'ufv_frame_consistency_residual', 'ufv_centers_consistency_residual'}
        backend = pathlib.Path(t3toolbox.__file__).resolve().parent / 'backend'
        uses = set()
        for path in sorted(backend.glob('*.py')):
            tree = ast.parse(path.read_text(), filename=str(path))
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                for node in ast.walk(function):
                    name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
                    if name in names:
                        uses.add((path.name, function.name, name))
        self.assertEqual(uses, {('ufv_operations.py', 'ufv_frame_consistency_residual', 'fv_frame_consistency_residual'),
                                ('ufv_operations.py', 'ufv_centers_consistency_residual', 'fv_centers_consistency_residual')})
        self.assertTrue(names <= set(fvo.__all__) | set(ufvo.__all__))


if __name__ == '__main__':
    unittest.main()
