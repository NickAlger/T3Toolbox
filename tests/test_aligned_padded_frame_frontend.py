# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The aligned padded frame's frontend twins (the aligned-padded-frame plan's T15 and the frontend half of T14 (h)).

- **Each twin is its backend call** (no added arithmetic; bitwise): ``T3Frame.padded`` / ``UT3Frame.padded``, the
  loop twins ``fv_aligned_padded_frame`` / ``ufv_aligned_padded_frame``, ``T3Tangent.align_padded`` /
  ``UT3Tangent.align_padded``, the tensors' ``oversampled_ranks``, and ``optimizers.newton_cg(padded_from=)``;
  stacks and sharing included.
- **The hint's class picks the projection**: a TuckerTensorTrain or a dense array on the ragged layer, a
  UniformTuckerTensorTrain with ``x_old``'s ``N`` on the uniform one; the other layer's hint, a wrong ``N``, and an
  unknown hint without ``project`` are structural errors naming the conversion.
- **The zero old point**: a rank-0 tensor and a partially-zero-rank one give the empty frame's start, on both layers,
  through the frame twins and through ``newton_cg(padded_from=)``.
- **The §3.6 validation messages** and the **safe-mode checks** (the same tensor, tied factors, an orthonormal start
  frame representing x0), which raise in safe mode and are skipped under ``safety.unsafe()``.

The twins' doctests run with the modules'. numpy-only (jax dispatch is in ``tests/test_dispatch.py``).
"""
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sgm
import t3toolbox.optimizers as topt
import t3toolbox.safety as safety
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.padded_frames as padded_frames
import t3toolbox.backend.uniform_padded_frames as uniform_padded_frames
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.ufv_conversions as ufc
import t3toolbox.backend.ufv_masking as ufv_masking
import t3toolbox.backend.tv_operations as tvo
import t3toolbox.backend.utv_operations as utvo
import t3toolbox.backend.uniform_fitting as uniform_fitting
import t3toolbox.backend.ranks as ranks

SHAPE, OLD, TARGET = (5, 6, 5), ((2, 2, 2), (1, 2, 2, 1)), ((3, 4, 3), (1, 3, 3, 1))
ZERO = ((0, 0, 0), (1, 0, 0, 1))
PARTIAL = ((0, 2, 2), (1, 0, 2, 1))


def equal_trees(a, b):
    """Bitwise equality of two (nested) tuples of arrays and static leaves."""
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_trees(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def points(seed=0):
    np.random.seed(seed)
    x_old = t3.TuckerTensorTrain.randn(SHAPE, *OLD)
    hint = t3.TuckerTensorTrain.randn(SHAPE, (5, 6, 5), (1, 5, 5, 1))
    return x_old, hint, ut3.UniformTuckerTensorTrain.from_t3(x_old), ut3.UniformTuckerTensorTrain.from_t3(hint)


class TestTwinsAreTheirBackendCalls(unittest.TestCase):

    def test_frame_padded(self):
        x_old, _, ux_old, _ = points()
        F = bvf.T3Frame.padded(x_old, *TARGET, seed=3)
        want = padded_frames.fv_padded_frame(fvc.t3_orthogonal_representations(x_old.data)[0], *TARGET, seed=3)
        self.assertTrue(equal_trees(F.data, want))
        UF, split = ubv.UT3Frame.padded(ux_old, *TARGET, seed=3)
        uwant, usplit = uniform_padded_frames.ufv_padded_frame(ufc.ut3_orthogonal_representations(
            uniform_fitting.uniform_minimal(ux_old).data)[0], *TARGET, seed=3)
        self.assertTrue(equal_trees(UF.data, uwant) and split == usplit)

    def test_frame_padded_stack_and_sharing(self):
        np.random.seed(1)
        x_stack = t3.TuckerTensorTrain.randn(SHAPE, *OLD, stack_shape=(2,))
        F = bvf.T3Frame.padded(x_stack, *TARGET)
        self.assertTrue(equal_trees(F.data, padded_frames.fv_padded_frame(
            fvc.t3_orthogonal_representations(x_stack.data)[0], *TARGET)))
        sharing = (0, 1, 0)
        x_shared = t3.TuckerTensorTrain.randn(SHAPE, *OLD).share(sharing, max_tucker_ranks=OLD[0], max_tt_ranks=OLD[1])
        F = bvf.T3Frame.padded(x_shared, *TARGET, sharing=sharing)
        self.assertTrue(equal_trees(F.data, padded_frames.fv_padded_frame(
            fvc.t3_orthogonal_representations(x_shared.data)[0], *TARGET, sharing=sharing)))
        self.assertTrue(np.array_equal(F.up_tucker_cores[0], F.up_tucker_cores[2]))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x_stack)
        targets = (np.array([[3, 3], [4, 3], [3, 3]]), np.array([[1, 1], [3, 3], [3, 3], [1, 1]]))
        UF, _ = ubv.UT3Frame.padded(ux, *targets)
        uwant, _ = uniform_padded_frames.ufv_padded_frame(ufc.ut3_orthogonal_representations(
            uniform_fitting.uniform_minimal(ux).data)[0], *targets)
        self.assertTrue(equal_trees(UF.data, uwant))

    def test_loop_twins(self):
        x_old, hint, ux_old, uhint = points()
        kw = dict(max_passes=4, tol=0.0)
        F, m = bvf.fv_aligned_padded_frame(x_old, *TARGET, hint, **kw)
        old = fvc.t3_orthogonal_representations(x_old.data)[0]
        Fb, mb = padded_frames.fv_aligned_padded_frame(old, *TARGET, hint.data, tvo.tv_project_t3_onto_tangent_space,
                                                       **kw)
        self.assertTrue(equal_trees(F.data, Fb) and equal_trees(m, mb))
        dense = hint.to_dense()
        F, m = bvf.fv_aligned_padded_frame(x_old, *TARGET, dense, **kw)
        Fb, mb = padded_frames.fv_aligned_padded_frame(old, *TARGET, dense, tvo.tv_project_dense_onto_tangent_space,
                                                       **kw)
        self.assertTrue(equal_trees(F.data, Fb) and equal_trees(m, mb))
        UF, um = ubv.ufv_aligned_padded_frame(ux_old, *TARGET, uhint, **kw)
        widths = ufv_masking.ufv_frame_widths(ux_old.N, 4, 3)
        UFb, umb = uniform_padded_frames.ufv_aligned_padded_frame(
            ufc.ut3_orthogonal_representations(uniform_fitting.uniform_minimal(ux_old).data)[0], *TARGET, uhint.data,
            utvo.utv_project_ut3_onto_tangent_space, widths, **kw)
        self.assertTrue(equal_trees(UF.data, UFb) and np.array_equal(um, umb))

    def test_align_padded(self):
        x_old, hint, ux_old, uhint = points()
        F = bvf.T3Frame.padded(x_old, *TARGET)
        v = t3m.MANIFOLD.project_ambient(F, hint)
        F1, mv = v.align_padded(*OLD)
        Fb, mvb = padded_frames.tv_align_padded_frame(F.data, v.variations.data, *OLD)
        self.assertTrue(equal_trees(F1.data, Fb) and np.array_equal(mv, mvb))
        UF, split = ubv.UT3Frame.padded(ux_old, *TARGET)
        uv = ut3m.UNIFORM_MANIFOLD.project_ambient(UF, uhint)
        UF1, umv = uv.align_padded(split, update='UL')
        UFb, umvb = uniform_padded_frames.utv_align_padded_frame(UF.data, uv.variations.data, split, update='UL')
        self.assertTrue(equal_trees(UF1.data, UFb) and np.array_equal(umv, umvb))

    def test_oversampled_ranks(self):
        x_old, _, ux_old, _ = points()
        self.assertEqual(x_old.oversampled_ranks(2, sharing=(0, 1, 0)),
                         ranks.oversampled_ranks(SHAPE, *OLD, 2, (0, 1, 0)))
        got = ux_old.oversampled_ranks(2)
        want = ranks.oversampled_ranks(SHAPE, np.asarray(ux_old.tucker_ranks), np.asarray(ux_old.tt_ranks), 2)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(got, want)))

    def test_newton_cg_padded_from(self):
        """The frontend newton_cg(padded_from=) is the backend call with the old point's (reduced) data; with no new
        keyword the frontend forwards none."""
        np.random.seed(2)
        ww = [np.random.randn(60, n) for n in SHAPE]
        data = t3.TuckerTensorTrain.randn(SHAPE, *TARGET).probe(ww)
        x_old = t3.TuckerTensorTrain.randn(SHAPE, *OLD)
        x0 = x_old.resize(SHAPE, *TARGET)
        for layer in ('ragged', 'uniform'):
            with self.subTest(layer=layer):
                geometry = t3m.MANIFOLD if layer == 'ragged' else ut3m.UNIFORM_MANIFOLD
                start = x0 if layer == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x0)
                old = x_old if layer == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x_old, N=start.N)
                _, front = topt.newton_cg(geometry, 'probe', ww, data, start, max_newton=3, padded_from=old,
                                          padded_max_passes=4)
                setup = topt.least_squares_problem(geometry, 'probe', ww, data, start)
                _, back = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=3, padded_from=old.data,
                                         padded_max_passes=4)
                self.assertEqual(front['losses'], back['losses'])
                self.assertEqual(front['start_movements'], back['start_movements'])
                _, plain = topt.newton_cg(geometry, 'probe', ww, data, start, max_newton=2)
                self.assertNotIn('start_frame', plain)


class TestHintDispatch(unittest.TestCase):

    def test_the_errors(self):
        x_old, hint, ux_old, uhint = points()
        with self.assertRaisesRegex(TypeError, 'to_t3'):
            bvf.fv_aligned_padded_frame(x_old, *TARGET, uhint)
        with self.assertRaisesRegex(TypeError, 'pass project='):
            bvf.fv_aligned_padded_frame(x_old, *TARGET, object())
        with self.assertRaisesRegex(ValueError, 'dense hint of shape'):
            bvf.fv_aligned_padded_frame(x_old, *TARGET, np.zeros((5, 6, 4)))
        with self.assertRaisesRegex(TypeError, 'from_t3'):
            ubv.ufv_aligned_padded_frame(ux_old, *TARGET, hint)
        with self.assertRaisesRegex(TypeError, 'pass project='):
            ubv.ufv_aligned_padded_frame(ux_old, *TARGET, hint.to_dense())
        wide = ut3.UniformTuckerTensorTrain.from_t3(hint, N=ux_old.N + 1)
        with self.assertRaisesRegex(ValueError, 'from_t3'):
            ubv.ufv_aligned_padded_frame(ux_old, *TARGET, wide)
        with self.assertRaisesRegex(TypeError, 'UniformTuckerTensorTrain.from_t3'):
            ubv.UT3Frame.padded(x_old, *TARGET)
        with self.assertRaisesRegex(TypeError, 'to_t3'):
            bvf.T3Frame.padded(ux_old, *TARGET)

    def test_a_custom_project_receives_raw_data(self):
        x_old, hint, ux_old, uhint = points()
        seen = []

        def project(frame_data, h):
            seen.append(type(frame_data).__name__)
            return tvo.tv_project_t3_onto_tangent_space(frame_data, h)
        bvf.fv_aligned_padded_frame(x_old, *TARGET, hint.data, project=project, max_passes=2, tol=0.0)
        self.assertEqual(set(seen), {'tuple'})


class TestZeroOldPoint(unittest.TestCase):
    """T14 (h), the frontend half."""

    def test_frame_twins(self):
        for layer in ('ragged', 'uniform'):
            with self.subTest(layer=layer):
                frames = []
                for old_ranks in (ZERO, PARTIAL):
                    x_old = t3.TuckerTensorTrain.zeros(SHAPE, *old_ranks)
                    if layer == 'ragged':
                        frames.append(bvf.T3Frame.padded(x_old, *TARGET).data)
                    else:
                        frames.append(ubv.UT3Frame.padded(ut3.UniformTuckerTensorTrain.from_t3(x_old, N=6), *TARGET)[0]
                                      .data)
                self.assertTrue(equal_trees(frames[0], frames[1]))
                empty = (padded_frames.fv_padded_frame(padded_frames.fv_empty_frame(SHAPE), *TARGET) if layer == 'ragged'
                         else uniform_padded_frames.ufv_padded_frame(uniform_padded_frames.ufv_empty_frame(SHAPE, 6),
                                                                     *TARGET)[0])
                self.assertTrue(equal_trees(frames[0], empty))

    def test_newton_cg(self):
        """newton_cg(padded_from=<rank 0>) and a partially-zero-rank padded_from start from the same frame."""
        np.random.seed(3)
        ww = [np.random.randn(60, n) for n in SHAPE]
        data = t3.TuckerTensorTrain.randn(SHAPE, *TARGET).probe(ww)
        x0 = t3.TuckerTensorTrain.zeros(SHAPE, *TARGET)
        for layer in ('ragged', 'uniform'):
            with self.subTest(layer=layer):
                geometry = t3m.MANIFOLD if layer == 'ragged' else ut3m.UNIFORM_MANIFOLD
                start = x0 if layer == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x0)
                frames = []
                for old_ranks in (ZERO, PARTIAL):
                    old = t3.TuckerTensorTrain.zeros(SHAPE, *old_ranks)
                    old = old if layer == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(old, N=start.N)
                    seen = []
                    topt.newton_cg(geometry, 'probe', ww, data, start, max_newton=1, padded_from=old,
                                   callback=lambda info: seen.append(info.lm.frame))
                    frames.append(seen[0])
                self.assertTrue(equal_trees(frames[0], frames[1]))


class TestNewtonValidation(unittest.TestCase):
    """The §3.6 messages, and the safe-mode checks (raised in safe mode, skipped under safety.unsafe())."""

    def setUp(self):
        np.random.seed(4)
        self.ww = [np.random.randn(60, n) for n in SHAPE]
        self.data = t3.TuckerTensorTrain.randn(SHAPE, *TARGET).probe(self.ww)
        self.x_old = t3.TuckerTensorTrain.randn(SHAPE, *OLD)
        self.x0 = self.x_old.resize(SHAPE, *TARGET)

    def run_newton(self, x0=None, geometry=t3m.MANIFOLD, **kw):
        return topt.newton_cg(geometry, 'probe', self.ww, self.data, self.x0 if x0 is None else x0, max_newton=1, **kw)

    def test_structural_messages(self):
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(self.x0)
        cases = (
            (dict(padded_from=ut3.UniformTuckerTensorTrain.from_t3(self.x_old)), TypeError, 'to_t3'),
            (dict(padded_from=t3.TuckerTensorTrain.randn((5, 6, 4), *OLD)), ValueError, 'has shape'),
            (dict(start_frame=ubv.ut3_orthogonal_representations(ux0)[0]), TypeError, 'must be a T3Frame'),
            (dict(geometry=t3m.COREWISE, padded_from=self.x_old), ValueError, 'manifold geometry'),
            (dict(geometry=t3m.COREWISE, g0_reference='cone'), ValueError, 'manifold geometry'),
            (dict(start_frame=bvf.t3_orthogonal_representations(self.x0)[0], padded_from=self.x_old), ValueError,
             'not both'),
            (dict(padded_tol=0.1), ValueError, 'without padded_from'),
            (dict(cone_tol=0.1), ValueError, 'cone reference is not computed'),
            (dict(padded_from=self.x_old, padded_max_passes=0, padded_tol=0.1), ValueError, 'padded_max_passes=0'),
        )
        for kw, error, message in cases:
            with self.subTest(**{k: type(v).__name__ for k, v in kw.items()}):
                with self.assertRaisesRegex(error, message):
                    self.run_newton(**kw)
        with self.subTest(uniform_N=True), self.assertRaisesRegex(ValueError, 'padded to N='):
            self.run_newton(x0=ux0, geometry=ut3m.UNIFORM_MANIFOLD,
                            padded_from=ut3.UniformTuckerTensorTrain.from_t3(self.x_old, N=ux0.N + 1))

    def test_safe_mode_checks(self):
        other = t3.TuckerTensorTrain.randn(SHAPE, *OLD)                      # not x0's tensor
        F = bvf.t3_orthogonal_representations(self.x0)[0]
        scaled = bvf.T3Frame(tuple(2.0 * U for U in F.up_tucker_cores), F.down_tt_cores, F.left_tt_cores,
                             F.right_tt_cores)                                 # not orthonormal
        np.random.seed(5)
        shared = (0, 1, 0)
        untied = t3.TuckerTensorTrain.randn(SHAPE, *OLD)
        cases = ((dict(padded_from=other), 'different tensors'), (dict(start_frame=scaled), 'not orthonormal'),
                 (dict(start_frame=bvf.t3_orthogonal_representations(other.resize(SHAPE, *TARGET))[0]),
                  'does not represent x0'),
                 (dict(geometry=sgm.shared(t3m.MANIFOLD, shared), padded_from=untied,
                       x0=untied.resize(SHAPE, *TARGET).share(shared, max_tucker_ranks=TARGET[0],
                                                               max_tt_ranks=TARGET[1])), 'tied'))
        for kw, message in cases:
            with self.subTest(check=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_newton(**kw)
                with safety.unsafe():
                    self.run_newton(**kw)


if __name__ == '__main__':
    unittest.main()
