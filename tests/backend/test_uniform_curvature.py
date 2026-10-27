# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""The exact Riemannian Hessian on the UNIFORM layer (S8): every uniform exact-mode operation against its
ragged twin, per stack element (numpy-only; jax dispatch is in ``tests/test_dispatch.py``).

The governing contract is the uniform layer's (``docs/uniform_equivalence_contract.md``): ``to_uniform -> op ->
to_ragged == op_ragged`` on the real blocks, padding don't-care on input. Dense-vs-ragged alone is blind to a
too-permissive mask (``docs/contributor/testing_strategy.md``), so each operation is checked three ways:

* **per element vs ragged** (<= 1e-13 relative) on stacks whose elements have DIFFERENT ranks (the variety),
  with padding forced on every family, at ``d`` = 1, 3 and 4, and with tangent stacks ``K``;
* **exact output masks**, derived non-circularly from the ragged frames' core shapes (never from a uniform
  mask): every slot outside the real extent is exactly zero;
* **garbage-padded inputs**: garbage in every padded slot of the frame, ``theta``, ``theta_hat``, ``z``, the
  residual and the sample (the boundary-bond slots of the corewise frame included) leaves the output
  bit-identical -- the directions are drawn as dense random supercores, so their padding is garbage in every
  per-element check too. A negative control shows the direction mask is load-bearing (F15 / F18).

Pieces (T17): ``ufv_lift_factors``; ``utv_corewise_lift[_transpose]`` (+ the adjoint identity on ``K`` and ``C``
stacks); the T3-T3 two-hole ``ut3_inner_product_corewise_*``; the probe-derivative two-hole
``ut3_probe_corewise_derivatives_hessian[_from_sweep]`` (+ chunked); ``UniformProbeDerivativesKind.corewise_hessian``
(T24: the theta_hat mask, the masks read off the FRAME at TT slack, T23 after the override); the uniform
geometries' exact members (``d = 1`` -> ``None``; shared -> the next slice's structural error); the backend exact
``LocalModel`` (hvp / curvature_hvp / quadratic / curvature_quadratic / gradient, ``K`` stacks) per element and,
through ``uniform_least_squares_problem``, against the ragged problem's dense actions; the Gauss-Newton forward and
hvp on a ``K`` stack (the S0c m1 gap); the uniform m27 (the exact gradient IS the Gauss-Newton one); T20 (flat
slots). End to end (T21 / T25): exact ``newton_cg`` from slack starts (reduced to full rank), uniform backend and
frontend row by row equal to the ragged runs; from a zero-padded restart (rank-deficient: the layers' frames
legitimately differ there) each converges to the ragged run's minimizer, and the frontend IS the backend run.
The full-length T21 / TT-slack runs are slow-tier (``T3TOOLBOX_SLOW_TESTS=1``).
"""
import dataclasses as dc
import os
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_linalg as t3la
import t3toolbox.backend.tv_operations as tvo
import t3toolbox.backend.ufv_conversions as ufc
import t3toolbox.backend.ufv_masking as ufm
import t3toolbox.backend.ufv_operations as ufo
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.ut3_conversions as ut3c
import t3toolbox.backend.ut3_linalg as ut3la
import t3toolbox.backend.ut3_masking as ut3mask
import t3toolbox.backend.ut3_operations as ut3o
import t3toolbox.backend.ut3_sampling as ut3s
import t3toolbox.backend.utv_operations as utvo
import t3toolbox.backend.utv_sampling as utvs

# Per case: the ragged structures of the stack elements (one -> C = (); two, of different ranks -> C = (2,)).
_CASES = {
    'd3': [((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))],
    'd3 varying': [((5, 6, 7), (2, 3, 2), (1, 2, 2, 1)), ((5, 6, 7), (3, 2, 3), (1, 3, 2, 1))],
    'd4 varying': [((4, 5, 4, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)), ((4, 5, 4, 5), (3, 2, 2, 3), (1, 3, 2, 3, 1))],
    'd1 varying': [((5,), (3,), (1, 1)), ((5,), (2,), (1, 1))],
    'tt slack': [((6, 6, 6), (4, 4, 4), (1, 9, 9, 1))],
}
_KS = [(), (2,)]
_TOL = 1e-13
_MANIFOLD, _COREWISE = bgeo.ManifoldGeometryOps(), bgeo.CorewiseGeometryOps()


# ---------------------------------------------------------------------------------------------------
# Packing: ragged per-element objects -> one uniform stack, padding FORCED (one more slot than any element)
# ---------------------------------------------------------------------------------------------------
def _unit_point(struct):
    """A random T3 scaled to ``||X||_HS = 1`` (well-scaled curvature)."""
    x = t3.TuckerTensorTrain.randn(*struct)
    nrm = float(np.linalg.norm(x.to_dense()))
    tk, tt = x.data
    return tuple(tk), tuple(tt[:-1]) + (tt[-1] / nrm,)


def _frame_dims(rframes):
    """Forced padded sizes for a list of ragged frames: (N, nU, nD, rL, rR), each one more than the max."""
    U = [u for f in rframes for u in f[0]]
    O = [o for f in rframes for o in f[1]]
    L = [g for f in rframes for g in f[2]]
    R = [g for f in rframes for g in f[3]]
    return (max(u.shape[-1] for u in U) + 1, max(u.shape[-2] for u in U) + 1, max(o.shape[-2] for o in O) + 1,
            max(max(g.shape[-3], g.shape[-1]) for g in L) + 1, max(max(g.shape[-3], g.shape[-1]) for g in R) + 1)


def _pack_frames(rframes, dims):
    N, nU, nD, rL, rR = dims
    packed = [ufc.t3frame_to_ut3frame(f, N=N, nU=nU, nD=nD, rL=rL, rR=rR) for f in rframes]
    return packed[0] if len(packed) == 1 else ufo.ufv_stack(tuple(packed), 4)


def _idx(k, c, n_elements):
    return tuple(k) + ((c,) if n_elements > 1 else ())


def _real_slices(sc, i, k, c, n_elements, shape):
    """The real block (shape ``shape``) of supercore ``sc`` at mode ``i`` and stack element ``(k, c)``."""
    return sc[(i,) + _idx(k, c, n_elements) + tuple(slice(0, s) for s in shape)]


def _ragged_slices(sc, K, n_elements, shapes_of):
    """Per element ``c``: the ragged ``K``-stacked family sliced out of a uniform supercore, with the real shapes
    ``shapes_of(c)`` (one per mode)."""
    out = []
    for c in range(n_elements):
        fam = []
        for i, shp in enumerate(shapes_of(c)):
            blocks = np.stack([_real_slices(sc, i, k, c, n_elements, shp) for k in np.ndindex(*K)]) if K else \
                _real_slices(sc, i, (), c, n_elements, shp)
            fam.append(blocks.reshape(tuple(K) + tuple(shp)))
        out.append(tuple(fam))
    return out


def _theta_shapes(rframe):
    """The real shapes of a raw direction at the base point (U, P): Udot (nU, N), Gdot (rL, nU, rL)."""
    return [u.shape[-2:] for u in rframe[0]], [g.shape[-3:] for g in rframe[2]]


def _var_shapes(rframe):
    """The real shapes of frame coordinates: V (nD, N), H (rL, nU, rR)."""
    return fvo.fv_variation_shapes(rframe)


def _garbage(sc, masked, scale=1e3, seed=0):
    """``sc`` with ``scale``-size noise added wherever ``masked`` (its masked twin) is exactly zero and ``sc`` is a
    padded slot -- the indicator is computed by masking a ones array, so it is the slot's padding, not its value."""
    rng = np.random.default_rng(seed)
    out = np.array(sc, copy=True, order='K')         # keep the memory layout: a garbage-robust result is BITWISE
    out += scale * (1.0 - masked) * rng.standard_normal(sc.shape)    # the clean one (a layout change could move
    return out                                                        # the BLAS rounding, not the values)


def _padding_indicator_ut3(sc_pair, shape, masks):
    ones = ut3mask.ut3_apply_masks((np.ones_like(sc_pair[0]), np.ones_like(sc_pair[1]), shape, masks))
    return tuple(np.asarray(o) for o in ones)


def _dirty_ut3(sc_pair, shape, masks, seed=0):
    """A supercore pair with garbage in every slot its masks call padding."""
    ind = _padding_indicator_ut3(sc_pair, shape, masks)
    return tuple(_garbage(s, m, seed=seed + j) for j, (s, m) in enumerate(zip(sc_pair, ind)))


def _dirty_frame(frame_data, seed=0):
    ones = ufm.ufv_apply_frame_masks(tuple(np.ones_like(s) for s in frame_data[:4]) + tuple(frame_data[4:]))
    return tuple(_garbage(s, m, seed=seed + j) for j, (s, m) in enumerate(zip(frame_data[:4], ones))) + \
        tuple(frame_data[4:])


def _compare(test, u, ragged_of, K, n_elements, label, tol=_TOL):
    """Per element vs ragged, and the exact output mask: ``u`` (a supercore ``(d,)+K+C+trail``) equals
    ``ragged_of(c)`` (a ``K``-stacked tuple of ``d`` ragged arrays) on each element's real blocks -- whose extents
    come from the RAGGED arrays' shapes -- and is exactly zero everywhere else."""
    u = np.asarray(u)
    ind = np.zeros(u.shape, bool)
    worst = 0.0
    for c in range(n_elements):
        fam = ragged_of(c)
        for k in np.ndindex(*K):
            num = den = 0.0                       # relative to the element's whole family (a gauged block can be
            for i, R in enumerate(fam):           # identically zero, its computed value pure roundoff)
                Rk = np.asarray(R)[k] if K else np.asarray(R)
                sl = (i,) + _idx(k, c, n_elements) + tuple(slice(0, s) for s in Rk.shape)
                ind[sl] = True
                num += float(np.sum((u[sl] - Rk) ** 2))
                den += float(np.sum(Rk ** 2))
            worst = max(worst, (num / max(den, 1e-300)) ** 0.5)
    test.assertTrue(np.all(u[~ind] == 0.0), '%s: a padded slot is not exactly zero' % label)
    test.assertLess(worst, tol, label)
    return worst


def _pair_relerr(a, b):
    """||a - b|| / ||b|| over a whole (Tucker, TT) supercore pair."""
    num = sum(float(np.sum((np.asarray(x) - np.asarray(y)) ** 2)) for x, y in zip(a, b))
    den = sum(float(np.sum(np.asarray(y) ** 2)) for y in b)
    return (num / den) ** 0.5


def _case(name, seed, K=()):
    """(ragged points, ragged frames, the packed uniform frame, dims, n_elements) for a case."""
    np.random.seed(seed)
    structs = _CASES[name]
    points = [_unit_point(s) for s in structs]
    rframes = [fvc.t3_orthogonal_representations(x)[0] for x in points]
    dims = _frame_dims(rframes)
    return points, rframes, _pack_frames(rframes, dims), dims, len(structs)


def _random_variation_sc(uframe, K, rng):
    """Dense random variation supercores (d,)+K+C+(...) -- garbage in the padding by construction."""
    tk_shape, tt_shape = ufo.ufv_variation_shapes(uframe)
    C = tuple(uframe[0].shape[1:-2])
    return (rng.standard_normal(tk_shape[:1] + tuple(K) + C + tk_shape[1 + len(C):]),
            rng.standard_normal(tt_shape[:1] + tuple(K) + C + tt_shape[1 + len(C):]))


def _random_theta_sc(uframe, K, rng):
    """Dense random raw directions shaped like the base point (U, P): (d,)+K+C+(nU, N) / (rL, nU, rL)."""
    U, L = uframe[0], uframe[2]
    return (rng.standard_normal(U.shape[:1] + tuple(K) + U.shape[1:]),
            rng.standard_normal(L.shape[:1] + tuple(K) + L.shape[1:]))


def _theta_data(uframe):
    return (uframe[0], uframe[2], tuple(uframe[4]), (uframe[5][0], uframe[5][2]))


# ---------------------------------------------------------------------------------------------------
class TestUniformLiftFactors(unittest.TestCase):
    """``ufv_lift_factors``: per element the ragged ``fv_lift_factors`` (pinvs, spectra, eps), the boundary
    pass-through slot, exactly-zero padding, the real-dimension clip, and garbage in the frame's padding ignored."""

    def test_per_element_vs_ragged(self):
        for name in _CASES:
            for cd in (0.0, 1e-3):
                with self.subTest(case=name, curvature_damping=cd):
                    _points, rframes, uframe, _dims, n = _case(name, 1)
                    lf = [fvo.fv_lift_factors(f, cd) for f in rframes]
                    ulf = ufo.ufv_lift_factors(uframe, cd)
                    _compare(self, ulf.tucker_pinvs, lambda c: lf[c].tucker_pinvs, (), n, 'tucker_pinvs')
                    _compare(self, ulf.bond_pinvs, lambda c: lf[c].bond_pinvs + (np.ones((1, 1)),), (), n,
                             'bond_pinvs + the boundary slot')
                    _compare(self, ulf.tucker_svals, lambda c: lf[c].tucker_svals, (), n, 'tucker_svals')
                    if len(lf[0].bond_svals):
                        _compare(self, ulf.bond_svals, lambda c: lf[c].bond_svals, (), n, 'bond_svals')
                    self.assertEqual(ulf.bond_svals.shape[0], len(rframes[0][0]) - 1)
                    eps = np.asarray(ulf.eps).reshape(-1)
                    self.assertTrue(np.allclose(eps, [float(l.eps) for l in lf], rtol=1e-13, atol=0.0))
                    dirty = ufo.ufv_lift_factors(_dirty_frame(uframe), cd)
                    for a, b in zip((ulf.tucker_pinvs, ulf.bond_pinvs, ulf.tucker_svals, ulf.bond_svals, ulf.eps),
                                    (dirty.tucker_pinvs, dirty.bond_pinvs, dirty.tucker_svals, dirty.bond_svals,
                                     dirty.eps)):
                        self.assertTrue(np.array_equal(a, b), 'garbage in the frame padding leaked')

    def test_zero_padded_restart_keeps_the_ragged_clip(self):
        """At a rank-deficient (zero-padded) point the clip decides which directions survive: the uniform lift
        clips each block at its REAL dimensions, so it keeps exactly the ragged directions (padded dims would
        loosen the clip)."""
        np.random.seed(3)
        x = t3.TuckerTensorTrain.randn((6, 6, 6), (2, 2, 2), (1, 2, 2, 1)).resize((6, 6, 6), (3, 3, 3), (1, 3, 3, 1))
        frame = fvc.t3_orthogonal_representations(x.data)[0]
        uframe = _pack_frames([frame], _frame_dims([frame]))
        for cd in (0.0, 1e-6):
            with self.subTest(curvature_damping=cd):
                lf, ulf = fvo.fv_lift_factors(frame, cd), ufo.ufv_lift_factors(uframe, cd)
                _compare(self, ulf.tucker_pinvs, lambda c: lf.tucker_pinvs, (), 1, 'tucker_pinvs', tol=1e-12)
                _compare(self, ulf.bond_pinvs, lambda c: lf.bond_pinvs + (np.ones((1, 1)),), (), 1, 'bond_pinvs',
                         tol=1e-12)
                self.assertEqual(float(bgeo._uniform_lift_sigma_min(ulf, uframe[5])),
                                 float(bgeo._lift_sigma_min(lf)))


    def test_the_clip_is_at_the_real_dimensions(self):
        """The clip decision itself: a point whose every factor has a second singular value at ~5e-16 -- 7.7e-16
        of the largest, above the ragged clip ``2 eps`` (2 x 2 blocks) and below a clip at the padded dimensions
        (``8 eps``, padding forced to 8) -- keeps that direction on both layers: the undamped pinv is ~1e15 in
        every slot, ragged and uniform alike (a clip at the padded dimensions would drop it, norm ~1)."""
        np.random.seed(0)
        shape = (5, 6, 7)
        x1, x2 = (t3.TuckerTensorTrain.randn(shape, (1, 1, 1), (1, 1, 1, 1)) for _ in range(2))
        x = x1 * (1.0 / float(np.linalg.norm(x1.to_dense()))) + x2 * (1e-15 / float(np.linalg.norm(x2.to_dense())))
        frame = fvc.t3_orthogonal_representations(x.data)[0]
        lf = fvo.fv_lift_factors(frame, 0.0)
        ulf = ufo.ufv_lift_factors(ufc.t3frame_to_ut3frame(frame, N=8, nU=8, nD=8, rL=8, rR=8), 0.0)
        for ragged, uniform in ((lf.tucker_pinvs, ulf.tucker_pinvs), (lf.bond_pinvs, ulf.bond_pinvs[:-1])):
            for P, U in zip(ragged, uniform):
                self.assertGreater(np.linalg.norm(P), 1e14)                     # kept by the ragged clip
                self.assertLess(abs(np.linalg.norm(U) / np.linalg.norm(P) - 1.0), 1e-2)


class TestUniformLift(unittest.TestCase):
    """``utv_corewise_lift[_transpose]``: per element the ragged lift / transpose, exact output masks (the base
    point's / the frame's variation extents), garbage-padded inputs ignored, and the adjoint identity per
    ``K + C`` element in the masked coordinate inner products."""

    def test_per_element_vs_ragged(self):
        rng = np.random.default_rng(5)
        for name in _CASES:
            for K in _KS:
                with self.subTest(case=name, K=K):
                    _points, rframes, uframe, _dims, n = _case(name, 6)
                    lf = [fvo.fv_lift_factors(f, 1e-3) for f in rframes]
                    ulf = ufo.ufv_lift_factors(uframe, 1e-3)
                    shape = tuple(uframe[4])
                    vmasks = ufm.ufv_variation_masks_over_stack(uframe[5], uframe[0].shape[1:-2],
                                                                tuple(K) + uframe[0].shape[1:-2])
                    v_sc = _random_variation_sc(uframe, K, rng)                   # garbage padding
                    v_r = [tuple(_ragged_slices(v_sc[j], K, n, lambda c, j=j: _var_shapes(rframes[c])[j])[c]
                                 for j in range(2)) for c in range(n)]
                    out = utvo.utv_corewise_lift(uframe, v_sc + (shape, vmasks), ulf)
                    ref = [tvo.tv_corewise_lift(rframes[c], v_r[c], lf[c]) for c in range(n)]
                    for j in range(2):
                        _compare(self, out[j], lambda c, j=j: ref[c][j], K, n, 'lift %d' % j)
                    clean = ufm.ufv_apply_variations_masks(v_sc + (shape, vmasks))
                    again = utvo.utv_corewise_lift(_dirty_frame(uframe), tuple(clean) + (shape, vmasks), ulf)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(out[:2], again[:2])))

                    y_sc = _random_theta_sc(uframe, K, rng)                       # raw, garbage padding
                    y_r = [tuple(_ragged_slices(y_sc[j], K, n, lambda c, j=j: _theta_shapes(rframes[c])[j])[c]
                                 for j in range(2)) for c in range(n)]
                    tout = utvo.utv_corewise_lift_transpose(uframe, y_sc + _theta_data(uframe)[2:], ulf)
                    tref = [tvo.tv_corewise_lift_transpose(rframes[c], y_r[c], lf[c]) for c in range(n)]
                    for j in range(2):
                        _compare(self, tout[j], lambda c, j=j: tref[c][j], K, n, 'transpose %d' % j)

    def test_adjoint_identity_on_K_and_C_stacks(self):
        rng = np.random.default_rng(7)
        for name in ('d3 varying', 'd4 varying', 'd1 varying'):
            for K in ((), (3,), (2, 2)):
                for cd in (0.0, 1e-2):
                    with self.subTest(case=name, K=K, curvature_damping=cd):
                        _points, rframes, uframe, _dims, n = _case(name, 8)
                        ulf = ufo.ufv_lift_factors(uframe, cd)
                        shape = tuple(uframe[4])
                        KC = tuple(K) + uframe[0].shape[1:-2]
                        vmasks = ufm.ufv_variation_masks_over_stack(uframe[5], uframe[0].shape[1:-2], KC)
                        v = _random_variation_sc(uframe, K, rng)
                        y = _random_theta_sc(uframe, K, rng)
                        Lv = utvo.utv_corewise_lift(uframe, v + (shape, vmasks), ulf)
                        LTy = utvo.utv_corewise_lift_transpose(uframe, y + _theta_data(uframe)[2:], ulf)
                        mv = ufm.ufv_apply_variations_masks(v + (shape, vmasks))
                        my = ut3mask.ut3_apply_masks(y + _theta_data(uframe)[2:])
                        axes = lambda a: (0,) + tuple(range(1 + len(KC), a.ndim))
                        lhs = sum(np.sum(a * b, axis=axes(a)) for a, b in zip(Lv[:2], my))
                        rhs = sum(np.sum(a * b, axis=axes(a)) for a, b in zip(mv, LTy[:2]))
                        self.assertEqual(np.shape(lhs), KC)                       # one identity per K + C element
                        self.assertLess(float(np.max(np.abs(lhs - rhs) / np.maximum(np.abs(lhs), 1e-300))), 1e-12)


class TestUniformInnerProductTwoHole(unittest.TestCase):
    """The uniform T3-T3 term (``ut3_inner_product_corewise_*``): per element the ragged one-hole / sweep /
    two-hole on a varying-rank stack, exact output masks, garbage in ``z``, ``theta`` and ``theta_hat`` ignored;
    and the direction mask is load-bearing (negative control)."""

    def _stack_ut3(self, cores_list, dims):
        N, n, r = dims
        uts = [ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*c), N=N, n=n, r=r) for c in cores_list]
        return uts[0].data if len(uts) == 1 else ut3.UniformTuckerTensorTrain.stack(uts).data

    def test_per_element_vs_ragged(self):
        rng = np.random.default_rng(9)
        for name in _CASES:
            for K in _KS:
                with self.subTest(case=name, K=K):
                    np.random.seed(10)
                    structs = _CASES[name]
                    n = len(structs)
                    thetas = [_unit_point(s) for s in structs]
                    zs = [_unit_point((s[0], tuple(r + 1 for r in s[1]), (1,) + tuple(r + 1 for r in s[2][1:-1]) + (1,)))
                          for s in structs]
                    dims = (max(structs[0][0]) + 1, max(max(st[1]) for st in structs) + 2, max(max(st[2]) for st in structs) + 2)
                    TH, Z = self._stack_ut3(thetas, dims), self._stack_ut3(zs, dims)
                    tk_shape, tt_shape = TH[0].shape, TH[1].shape
                    D = (rng.standard_normal(tk_shape[:1] + tuple(K) + tk_shape[1:]),
                         rng.standard_normal(tt_shape[:1] + tuple(K) + tt_shape[1:]))
                    shapes = lambda c: ([u.shape for u in thetas[c][0]], [g.shape for g in thetas[c][1]])
                    D_r = [tuple(_ragged_slices(D[j], K, n, lambda cc, j=j: shapes(cc)[j])[c] for j in range(2))
                           for c in range(n)]
                    out = ut3la.ut3_inner_product_corewise_hessian(Z, TH, D)
                    for j in range(2):
                        _compare(self, out[j],
                                 lambda c, j=j: t3la.t3_inner_product_corewise_hessian(zs[c], thetas[c], D_r[c])[j],
                                 K, n, 'two-hole %d' % j)
                    grad = ut3la.ut3_inner_product_corewise_transpose(Z, TH)
                    for j in range(2):
                        _compare(self, grad[j], lambda c, j=j: t3la.t3_inner_product_corewise_transpose(
                            zs[c], thetas[c])[j], (), n, 'one-hole %d' % j)
                    dZ = _dirty_ut3(Z[:2], Z[2], Z[3], seed=1) + Z[2:]
                    dTH = _dirty_ut3(TH[:2], TH[2], TH[3], seed=2) + TH[2:]
                    again = ut3la.ut3_inner_product_corewise_hessian_from_sweep(
                        dZ, dTH, D, ut3la.ut3_precompute_inner_product_sweep(dZ, dTH))
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(out, again)), 'garbage leaked')
                    if n == 1 and not K and len(structs[0][0]) > 1:      # negative control: the mask matters
                        bare = t3la.t3_inner_product_corewise_hessian_from_sweep(
                            ut3mask.ut3_apply_masks(Z), ut3mask.ut3_apply_masks(TH), D,
                            ut3la.ut3_precompute_inner_product_sweep(Z, TH))
                        masked = ut3mask.ut3_apply_masks(bare + TH[2:])
                        self.assertFalse(np.allclose(masked[1], out[1]), 'an unmasked direction went unnoticed')


# ---------------------------------------------------------------------------------------------------
# The probe-derivative two-hole
# ---------------------------------------------------------------------------------------------------
def _pd_sample(shape, W, order, n, rng):
    ww = [rng.standard_normal((W, N)) for N in shape]
    pp = [rng.standard_normal((W, N)) for N in shape]
    C = (n,) if n > 1 else ()
    zt = [rng.standard_normal((order + 1, W) + C + (N,)) for N in shape]
    return ww, pp, zt


def _dirty_packed_vectors(packed, shape, seed):
    """Garbage in the padded mode slots (beyond N_i) of a packed (d,)+...+(N,) array."""
    sm = np.arange(packed.shape[-1]) < np.asarray(shape)[:, None]               # (d, N)
    ind = sm.reshape((packed.shape[0],) + (1,) * (packed.ndim - 2) + (packed.shape[-1],))
    return _garbage(packed, np.broadcast_to(ind, packed.shape).astype(float), seed=seed)


class TestUniformProbeDerivativesTwoHole(unittest.TestCase):
    """``ut3_probe_corewise_derivatives_hessian[_from_sweep]``: per element the ragged two-hole (orders 0-2,
    ``K`` stacks, ``d`` = 1, 3, 4, varying ranks, dense and W-chunked), exact output masks, and garbage in the
    padding of theta, theta_hat, the residual jets and the probe / perturbation vectors ignored; the
    sweep-reusing form equals the one-shot form."""

    def test_per_element_vs_ragged(self):
        rng = np.random.default_rng(11)
        W = 7
        for name in _CASES:
            for K in _KS:
                for order, chunk in ((0, None), (2, None), (1, 3)):
                    with self.subTest(case=name, K=K, order=order, chunk_size=chunk):
                        _points, rframes, uframe, _dims, n = _case(name, 12)
                        shape = tuple(uframe[4])
                        ww, pp, zt = _pd_sample(shape, W, order, n, rng)
                        theta = _theta_data(uframe)
                        D = _random_theta_sc(uframe, K, rng)
                        D_r = [tuple(_ragged_slices(D[j], K, n, lambda cc, j=j: _theta_shapes(rframes[cc])[j])[c]
                                     for j in range(2)) for c in range(n)]
                        out = ut3s.ut3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, D, order, chunk_size=chunk)
                        ref = []
                        for c in range(n):
                            ztc = [z[..., c, :] for z in zt] if n > 1 else zt
                            ref.append(sd.t3_probe_corewise_derivatives_hessian(
                                ztc, ww, pp, (rframes[c][0], rframes[c][2]), D_r[c], order, chunk_size=chunk))
                        for j in range(2):
                            _compare(self, out[j], lambda c, j=j: ref[c][j], K, n, 'two-hole %d' % j)
                        # garbage everywhere padded: theta (the frame), the jets and vectors (beyond N_i)
                        N = uframe[0].shape[-1]
                        dirty_theta = _dirty_ut3(theta[:2], theta[2], theta[3], seed=3) + theta[2:]
                        pk = lambda v, s: _dirty_packed_vectors(ut3o.pack_if_ragged(v, N), shape, s)
                        again = ut3s.ut3_probe_corewise_derivatives_hessian(
                            pk(zt, 4), pk(ww, 5), pk(pp, 6), dirty_theta, D, order, chunk_size=chunk)
                        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(out, again)), 'garbage leaked')
                        sweep = utvs.utv_precompute_apply_frame_sweep_jets(ufc.ut3_corewise_frame(theta), ww, pp,
                                                                           order)
                        reuse = ut3s.ut3_probe_corewise_derivatives_hessian_from_sweep(zt, theta, D, order, sweep,
                                                                                        chunk_size=chunk)
                        self.assertLess(_pair_relerr(reuse, out), _TOL)

    def test_the_direction_mask_is_load_bearing(self):
        """Negative control (F15 / F18): the kernel on the masked theta but the UNmasked direction corrupts the
        real block -- what the wrapper's second mask prevents."""
        rng = np.random.default_rng(13)
        _points, rframes, uframe, _dims, n = _case('d3', 14)
        shape, order, W = tuple(uframe[4]), 2, 6
        ww, pp, zt = _pd_sample(shape, W, order, n, rng)
        theta = _theta_data(uframe)
        D = _random_theta_sc(uframe, (), rng)
        good = ut3s.ut3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, D, order)
        N = uframe[0].shape[-1]
        pk = lambda v: ut3o.pack_if_ragged(v, N)
        bare = sd.t3_probe_corewise_derivatives_hessian(pk(zt), pk(ww), pk(pp), ut3mask.ut3_apply_masks(theta), D,
                                                        order)
        bare = ut3mask.ut3_apply_masks(bare + theta[2:])                       # even re-masked, the real block is off
        self.assertGreater(np.linalg.norm(bare[1] - good[1]), 1e-3 * np.linalg.norm(good[1]))


# ---------------------------------------------------------------------------------------------------
# The uniform kind's corewise_hessian (T24, T23 after the override)
# ---------------------------------------------------------------------------------------------------
def _tt_slack_uniform(seed):
    """A TT-slack point (non-minimal: its frame's left ranks [1,4,9,1] < the point's [1,9,9,1]), ragged and as a
    uniform point, and its RAGGED frame packed at the point's padded sizes -- so the frame's (up, left) masks are a
    strict subset of the point's (F23) and its coordinates are the ragged frame's slot for slot."""
    np.random.seed(seed)
    x = t3.TuckerTensorTrain.randn((6, 6, 6), (4, 4, 4), (1, 9, 9, 1))
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    rframe = fvc.t3_orthogonal_representations(x.data)[0]
    n, r = ux.data[0].shape[-2], ux.data[1].shape[-1]
    return x, ux, rframe, ufc.t3frame_to_ut3frame(rframe, N=ux.N, nU=n, nD=n, rL=r, rR=r)


class TestUniformKindCorewiseHessian(unittest.TestCase):
    """``UniformProbeDerivativesKind.corewise_hessian``: it replaces the ``_UniformKind`` stub (T23); its masks are
    the FRAME's ``(up, left)`` (m29: at TT slack ``frame[5]`` is a strict subset of ``self.masks``, and the
    point masks would let the frame's padded completions into theta); T24: theta masked == the sweep's
    ``(mb[0], mb[2])``, and garbage in theta_hat's padded slots (every edge) == zeroed padding, the real block
    bitwise."""

    def test_the_override_replaces_the_stub(self):
        self.assertIsNot(uf.UniformProbeDerivativesKind.corewise_hessian, uf._UniformKind.corewise_hessian)
        self.assertFalse(getattr(uf.UniformProbeDerivativesKind.corewise_hessian, '__isabstractmethod__', False))
        self.assertNotIn('UniformProbeDerivativesKind', uf._UNIFORM_HESSIAN_SLICE)
        mro = uf.UniformProbeDerivativesKind.__mro__
        self.assertLess(mro.index(uf._UniformKind), mro.index(bfit.ProbeDerivativesKind))   # still precedes

    def test_masks_from_the_frame_at_tt_slack(self):
        rng = np.random.default_rng(15)
        x, ux, rframe, frame = _tt_slack_uniform(16)
        point_up, point_left = ux.data[3]
        self.assertTrue(np.all(frame[5][2] <= point_left) and not np.array_equal(frame[5][2], point_left))   # F23
        order, W = 1, 8
        shape = x.shape
        ww = [rng.standard_normal((W, N)) for N in shape]
        pp = [rng.standard_normal((W, N)) for N in shape]
        kind = uf.UniformProbeDerivativesKind.from_point(ux.data, order=order, weight=[1.0, 0.5], chunk_size=None)
        sample = uf.pack_sample('probe_derivatives', (ww, pp), ux.N)
        sweep = kind.precompute(frame, sample)
        r = [rng.standard_normal((order + 1, W, N)) for N in shape]
        rp = uf.pack_data('probe_derivatives', r, ux.N)
        base = (frame[0], frame[2])
        for K in _KS:
            with self.subTest(K=K):
                D = _random_theta_sc(frame, K, rng)
                out = kind.corewise_hessian(D, rp, sample, frame, sweep, base)
                D_r = tuple(_ragged_slices(D[j], K, 1, lambda c, j=j: _theta_shapes(rframe)[j])[0] for j in range(2))
                ragged_kind = bfit.ProbeDerivativesKind(order=order, weight=[1.0, 0.5], chunk_size=None)
                rsweep = ragged_kind.precompute(rframe, (ww, pp))
                ref = ragged_kind.corewise_hessian(D_r, r, (ww, pp), rframe, rsweep, (rframe[0], rframe[2]))
                for j in range(2):                         # the extent is the FRAME's (the ragged frame's shapes)
                    _compare(self, out[j], lambda c, j=j: ref[j], K, 1, 'kind %d' % j)
                wrong = ut3s.ut3_probe_corewise_derivatives_hessian_from_sweep(        # the point's masks instead
                    kind._apply_weight(rp, 2), (base[0], base[1], ux.data[2], ux.data[3]), D, order, sweep,
                    chunk_size=None)
                self.assertFalse(np.allclose(wrong[1], out[1]), 'the point masks gave the same answer')

    def test_theta_hat_masking_unit(self):
        """T24 (F18, F25): masked theta == the sweep's (mb[0], mb[2]); garbage in theta_hat's padded slots (every
        edge padded) gives the zeroed-padding answer, the real block bitwise."""
        rng = np.random.default_rng(17)
        _points, rframes, uframe, _dims, n = _case('d3 varying', 18)
        shape, order, W = tuple(uframe[4]), 1, 5
        masks = (uframe[5][0], uframe[5][2])
        for m in masks:                                                  # every edge has padding
            self.assertTrue(np.all(~m[..., -1]))
        kind = uf.UniformProbeDerivativesKind.from_point((None, None, shape, masks), order=order)
        ww, pp, _zt = _pd_sample(shape, W, order, n, rng)
        sample = uf.pack_sample('probe_derivatives', (ww, pp), uframe[0].shape[-1])
        sweep = kind.precompute(uframe, sample)
        theta = ut3mask.ut3_apply_masks(_theta_data(uframe))
        self.assertTrue(np.array_equal(theta[0], sweep[0][0]) and np.array_equal(theta[1], sweep[0][2]))
        r = rng.standard_normal((len(shape), order + 1, W, n, uframe[0].shape[-1]))
        D = _random_theta_sc(uframe, (2,), rng)
        clean = ut3mask.ut3_apply_masks(D + (shape, masks))
        a = kind.corewise_hessian(D, r, sample, uframe, sweep, (uframe[0], uframe[2]))
        b = kind.corewise_hessian(tuple(clean), r, sample, uframe, sweep, (uframe[0], uframe[2]))
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertGreater(min(np.abs(np.asarray(D[j]) - np.asarray(clean[j])).max() for j in range(2)), 1.0)


# ---------------------------------------------------------------------------------------------------
# The uniform geometries' exact members and the exact LocalModel
# ---------------------------------------------------------------------------------------------------
def _uniform_exact_model(uframe, plain_masks, kind, sample, residual, curvature_damping=None, geom=None):
    """A backend exact LocalModel AT A GIVEN uniform frame (``Problem.local_model`` builds its own frame, which is
    only gauge-equivalent to the ragged one): so the coordinates match the ragged model's slot for slot."""
    geom = geom or bgeo.UniformManifoldGeometryOps.from_frame(uframe, plain_masks)
    sweep = kind.precompute(uframe, sample)
    mg = geom.project(uframe, kind.transpose(residual, sample, uframe, sweep))
    cd = geom.curvature_precompute(uframe, mg, None, curvature_damping)
    return bopt.LocalModel(geom, kind, sample, uframe, sweep, residual, kind.w_axes(sample), None, None,
                           bopt.CurvatureState(mg, cd))


def _ragged_exact_model(rframe, kind, sample, residual, curvature_damping=None, geom=_MANIFOLD):
    sweep = kind.precompute(rframe, sample)
    mg = geom.project(rframe, kind.transpose(residual, sample, rframe, sweep))
    cd = geom.curvature_precompute(rframe, mg, None, curvature_damping)
    return bopt.LocalModel(geom, kind, sample, rframe, sweep, residual, kind.w_axes(sample), None, None,
                           bopt.CurvatureState(mg, cd))


class TestUniformGeometryExactMembers(unittest.TestCase):
    """The uniform geometries' four exact-mode members per element vs the ragged geometry's, on a frame packed
    from the ragged one: the curvature data (``grad f`` as a doubled-rank T3, densely; the real-slot sigma_min;
    eps), the lift, its transpose and the tangential term; ``d = 1`` gives ``None``; a shared uniform geometry
    raises the next slice's structural error, before any work; the corewise geometry's are the identity / ``()``
    / ``None`` and reject a damping."""

    def test_manifold_members_per_element(self):
        rng = np.random.default_rng(19)
        for name in ('d3', 'd3 varying', 'd4 varying', 'tt slack'):
            for K in _KS:
                with self.subTest(case=name, K=K):
                    _points, rframes, uframe, _dims, n = _case(name, 20)
                    geom = bgeo.UniformManifoldGeometryOps.from_frame(uframe, (uframe[5][0], uframe[5][2]))
                    shape = tuple(uframe[4])
                    mg_sc = geom.project(uframe, _random_variation_sc(uframe, (), rng))
                    mg_r = [tuple(_ragged_slices(mg_sc[j], (), n, lambda c, j=j: _var_shapes(rframes[c])[j])[c]
                                  for j in range(2)) for c in range(n)]
                    cd_u = geom.curvature_precompute(uframe, mg_sc, None, 1e-3)
                    cd_r = [_MANIFOLD.curvature_precompute(rframes[c], mg_r[c], None, 1e-3) for c in range(n)]
                    Zd = np.asarray(ut3c.ut3_to_dense(cd_u.z))
                    for c in range(n):
                        Zr = np.asarray(t3.TuckerTensorTrain(*cd_r[c].z).to_dense())
                        Zc = Zd[c] if n > 1 else Zd
                        self.assertLess(np.linalg.norm(Zc - Zr) / np.linalg.norm(Zr), _TOL)
                    sig = np.asarray(cd_u.sigma_min).reshape(-1)
                    self.assertTrue(np.allclose(sig, [float(r.sigma_min) for r in cd_r], rtol=_TOL, atol=0.0))
                    self.assertTrue(np.allclose(np.asarray(cd_u.eps).reshape(-1), [float(r.eps) for r in cd_r],
                                                rtol=1e-13, atol=0.0))
                    v = _random_variation_sc(uframe, K, rng)
                    v_r = [tuple(_ragged_slices(v[j], K, n, lambda cc, j=j: _var_shapes(rframes[cc])[j])[c]
                                 for j in range(2)) for c in range(n)]
                    th = geom.lift(uframe, v, cd_u)
                    th_r = [_MANIFOLD.lift(rframes[c], v_r[c], cd_r[c]) for c in range(n)]
                    for j in range(2):
                        _compare(self, th[j], lambda c, j=j: th_r[c][j], K, n, 'lift %d' % j)
                    D = _random_theta_sc(uframe, K, rng)
                    D_r = [tuple(_ragged_slices(D[j], K, n, lambda cc, j=j: _theta_shapes(rframes[cc])[j])[c]
                                 for j in range(2)) for c in range(n)]
                    kt = geom.tangential_curvature(uframe, D, cd_u)
                    kt_r = [_MANIFOLD.tangential_curvature(rframes[c], D_r[c], cd_r[c]) for c in range(n)]
                    for j in range(2):
                        _compare(self, kt[j], lambda c, j=j: kt_r[c][j], K, n, 'tangential %d' % j)
                    lt = geom.lift_transpose(uframe, D, cd_u)
                    lt_r = [_MANIFOLD.lift_transpose(rframes[c], D_r[c], cd_r[c]) for c in range(n)]
                    for j in range(2):
                        _compare(self, lt[j], lambda c, j=j: lt_r[c][j], K, n, 'lift_transpose %d' % j)

    def test_the_frame_masks_at_tt_slack_through_the_model(self):
        """m29 for the GEOMETRY members too (S8 review): at TT slack the kind and the geometry hold the POINT's masks
        (``self.masks``, a strict superset of the frame's, F23) while the frame carries GARBAGE in every slot outside
        its own extent -- including the slots between the frame's and the point's extents, where a uniform frame
        built at such a point keeps its completions. Every exact action is then bit-identical to the clean-frame
        model and, per element, the ragged model's at the slack frame. (A ``tangential_curvature`` masking theta with
        ``self.masks`` passes every clean-frame check and fails here.)"""
        rng = np.random.default_rng(41)
        x, ux, rframe, frame = _tt_slack_uniform(42)
        shape, order, W = x.shape, 1, 6
        ww = [rng.standard_normal((W, N)) for N in shape]
        pp = [rng.standard_normal((W, N)) for N in shape]
        r = [rng.standard_normal((order + 1, W, N)) for N in shape]
        kind = uf.UniformProbeDerivativesKind.from_point(ux.data, order=order, weight=[1.0, 0.5], chunk_size=None)
        sample = uf.pack_sample('probe_derivatives', (ww, pp), ux.N)
        rp = uf.pack_data('probe_derivatives', r, ux.N)
        geom = bgeo.UniformManifoldGeometryOps.from_frame(frame, ux.data[3])          # held masks: the POINT's
        self.assertFalse(np.array_equal(geom.masks[1], frame[5][2]))
        clean = _uniform_exact_model(frame, ux.data[3], kind, sample, rp, 1e-5, geom=geom)
        dirty = _uniform_exact_model(_dirty_frame(frame, seed=43), ux.data[3], kind, sample, rp, 1e-5, geom=geom)
        rkind = bfit.ProbeDerivativesKind(order=order, weight=[1.0, 0.5], chunk_size=None)
        ragged = _ragged_exact_model(rframe, rkind, (ww, pp), r, 1e-5)
        for K in _KS:
            with self.subTest(K=K):
                p = _random_variation_sc(frame, K, rng)
                p_r = tuple(_ragged_slices(p[j], K, 1, lambda c, j=j: _var_shapes(rframe)[j])[0] for j in range(2))
                for label, act in (('hvp', lambda m, q: m.hvp(q)), ('curvature_hvp', lambda m, q: m.curvature_hvp(q))):
                    out = act(dirty, p)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(out, act(clean, p))), label)
                    ref = act(ragged, p_r)
                    for j in range(2):
                        _compare(self, out[j], lambda c, j=j: ref[j], K, 1, '%s %d' % (label, j), tol=1e-12)
                self.assertTrue(np.array_equal(np.asarray(dirty.curvature_quadratic(p)),
                                               np.asarray(clean.curvature_quadratic(p))))

    def test_d1_and_shared_and_corewise(self):
        rng = np.random.default_rng(21)
        _points, rframes, uframe, _dims, _n = _case('d1 varying', 22)
        geom = bgeo.UniformManifoldGeometryOps.from_frame(uframe, (uframe[5][0], uframe[5][2]))
        mg = geom.project(uframe, _random_variation_sc(uframe, (), rng))
        self.assertIsNone(geom.curvature_precompute(uframe, mg, None, None))           # flat: d = 1
        with self.assertRaisesRegex(ValueError, 'curvature_damping must be >= 0'):
            geom.curvature_precompute(uframe, mg, None, -1.0)
        np.random.seed(23)
        x = t3.TuckerTensorTrain.randn((5, 5, 6), (2, 2, 2), (1, 2, 2, 1))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        for base in (bgeo.UniformManifoldGeometryOps.from_point(ux.data),
                     bgeo.UniformCorewiseGeometryOps.from_point(ux.data)):
            with self.subTest(shared=type(base).__name__):
                shared = base.with_sharing((0, 0, 1))
                f = shared.frame((ux.data[0], ux.data[1]))
                with self.assertRaisesRegex(NotImplementedError, 'lands in slice S9'):
                    shared.curvature_precompute(f, None, None, None)
        cgeo = bgeo.UniformCorewiseGeometryOps.from_point(ux.data)
        f = cgeo.frame((ux.data[0], ux.data[1]))
        self.assertEqual(cgeo.curvature_precompute(f, None, None, None), ())
        with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
            cgeo.curvature_precompute(f, None, None, 1e-6)
        v = (ux.data[0], ux.data[1])
        self.assertIs(cgeo.lift(f, v, ()), v)
        self.assertIs(cgeo.lift_transpose(f, v, ()), v)
        self.assertIsNone(cgeo.tangential_curvature(f, v, ()))


def _exact_models(name, seed, rng, geometry='manifold', order=1, W=7, weight=(1.0, 0.5)):
    """(uniform exact model at a frame packed from the ragged ones, the ragged exact models per element,
    the uniform frame, the per-element variation shapes, n_elements): residual and sample garbage-padded."""
    _points, rframes, uframe, dims, n = _case(name, seed)
    shape = tuple(uframe[4])
    ww, pp, _ = _pd_sample(shape, W, order, n, rng)
    r = [rng.standard_normal((order + 1, W) + ((n,) if n > 1 else ()) + (N,)) for N in shape]
    N = uframe[0].shape[-1]
    rp = _dirty_packed_vectors(ut3o.pack_if_ragged(r, N), shape, 31)        # garbage beyond N_i
    sample = (_dirty_packed_vectors(ut3o.pack_if_ragged(ww, N), shape, 32),
              _dirty_packed_vectors(ut3o.pack_if_ragged(pp, N), shape, 33))
    masks = (uframe[5][0], uframe[5][2])
    kind = uf.UniformProbeDerivativesKind.from_point((None, None, shape, masks), order=order,
                                                     weight=list(weight), chunk_size=None)
    rkind = bfit.ProbeDerivativesKind(order=order, weight=list(weight), chunk_size=None)
    if geometry == 'manifold':
        lm = _uniform_exact_model(uframe, masks, kind, sample, rp, 1e-4)
        rlm = [_ragged_exact_model(rframes[c], rkind, (ww, pp), [z[..., c, :] for z in r] if n > 1 else r, 1e-4)
               for c in range(n)]
        vshapes = lambda c: _var_shapes(rframes[c])
    else:
        cframe = ufc.ut3_corewise_frame(_theta_data(uframe))
        geom = bgeo.UniformCorewiseGeometryOps(shape, masks)
        lm = _uniform_exact_model(cframe, masks, kind, sample, rp, None, geom=geom)
        rcf = [fvc.t3_corewise_frame((f[0], f[2])) for f in rframes]
        rlm = [_ragged_exact_model(rcf[c], rkind, (ww, pp), [z[..., c, :] for z in r] if n > 1 else r, None,
                                   geom=_COREWISE) for c in range(n)]
        vshapes = lambda c: _var_shapes(rcf[c])
        uframe = cframe
    return lm, rlm, uframe, vshapes, n


class TestUniformExactLocalModel(unittest.TestCase):
    """The backend exact ``LocalModel`` on the uniform layer, at a frame packed from the ragged one: every exact
    action per element (``K`` stacks of trial tangents, garbage-padded) equal to the ragged model's; the uniform
    m27 (the exact gradient IS the Gauss-Newton gradient, bit for bit); T20 (flat slots: ``d = 1`` -> no
    curvature data and ``hvp ==`` Gauss-Newton; maximal rank -> the normal curvature vanishes)."""

    def test_every_exact_action_per_element(self):
        rng = np.random.default_rng(25)
        for geometry in ('manifold', 'corewise'):
            for name in ('d3', 'd3 varying', 'd4 varying'):
                for K in _KS:
                    with self.subTest(geometry=geometry, case=name, K=K):
                        lm, rlm, uframe, vshapes, n = _exact_models(name, 26, rng, geometry)
                        p = _random_variation_sc(uframe, K, rng)
                        p_r = [tuple(_ragged_slices(p[j], K, n, lambda cc, j=j: vshapes(cc)[j])[c] for j in range(2))
                               for c in range(n)]
                        for label, act in (('hvp', lambda m, q: m.hvp(q)),
                                           ('curvature_hvp', lambda m, q: m.curvature_hvp(q))):
                            out = act(lm, p)
                            refs = [act(rlm[c], p_r[c]) for c in range(n)]
                            for j in range(2):
                                _compare(self, out[j], lambda c, j=j: refs[c][j], K, n, '%s %d' % (label, j))
                        for label, act in (('quadratic', lambda m, q: m.quadratic(q)),
                                           ('curvature_quadratic', lambda m, q: m.curvature_quadratic(q))):
                            got = np.asarray(act(lm, p))
                            self.assertEqual(got.shape, tuple(K) + ((n,) if n > 1 else ()))
                            for c in range(n):
                                want = np.asarray(act(rlm[c], p_r[c]))
                                g = got[..., c] if n > 1 else got
                                self.assertLess(float(np.max(np.abs(g - want) / np.abs(want))), _TOL, label)
                        for j in range(2):
                            _compare(self, lm.gradient[j], lambda c, j=j: rlm[c].gradient[j], (), n, 'gradient')

    def test_garbage_in_the_frame_padding(self):
        """Garbage in EVERY padded slot of the frame's four supercores -- on the corewise frame ``(U, G, G, G)`` that
        includes the point's padded boundary-bond slots, the ones the boundary squash sums over -- leaves every
        exact action bit-identical, on both uniform geometries (and with garbage-padded residual, sample and
        trial tangent, as everywhere in this class)."""
        rng = np.random.default_rng(37)
        for geometry in ('manifold', 'corewise'):
            with self.subTest(geometry=geometry):
                lm, _rlm, uframe, _vs, n = _exact_models('d3 varying', 38, rng, geometry)
                dirty = _dirty_frame(uframe, seed=39)
                if geometry == 'corewise':                     # (U, G, G, G): ONE garbage-padded G, three times
                    dirty = (dirty[0], dirty[2], dirty[2], dirty[2]) + tuple(uframe[4:])
                    left = uframe[5][2]
                    self.assertTrue(np.all(~left[0][..., 1:]) and np.all(~left[-1][..., 1:]))   # the boundary pads
                    self.assertGreater(float(np.abs(dirty[2][0][..., 1:, :, :]).max()), 1.0)    # ... hold garbage
                lmd = _uniform_exact_model(dirty, (uframe[5][0], uframe[5][2]), lm.kind, lm.sample, lm.residual,
                                           1e-4 if geometry == 'manifold' else None,
                                           geom=None if geometry == 'manifold' else lm.geom)
                p = _random_variation_sc(uframe, (2,), rng)
                for act in (lambda m: m.hvp(p), lambda m: m.curvature_hvp(p), lambda m: m.gradient,
                            lambda m: (m.quadratic(p), m.curvature_quadratic(p))):
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(act(lm), act(lmd))))

    def test_m27_the_exact_gradient_is_the_gauss_newton_gradient(self):
        """Through ``uniform_least_squares_problem``: the exact model's gradient is the Gauss-Newton model's, bit
        for bit, on both uniform geometries (the misfit transpose is done once, by the same expression)."""
        rng = np.random.default_rng(27)
        np.random.seed(28)
        x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        ww, pp, _ = _pd_sample(x.shape, 9, 2, 1, rng)
        data = sd.t3_probe_derivatives(ww, pp, _unit_point(((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))), 2)
        for geometry in ('manifold', 'corewise'):
            with self.subTest(geometry=geometry):
                prob = uf.uniform_least_squares_problem(geometry, 'probe_derivatives', ux, (ww, pp), data, order=2,
                                                        weight=[1.0, 0.5, 0.25])
                sc = (ux.data[0], ux.data[1])
                gn, ex = prob.local_model(sc), prob.local_model(sc, hessian='exact')
                self.assertTrue(ex.is_exact)
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(gn.gradient, ex.gradient)))

    def test_problem_path_matches_the_ragged_problem_densely(self):
        """``uniform_least_squares_problem(...).local_model(hessian='exact')`` builds its own (gauge-equivalent)
        frame, so compare GEOMETRY: the dense ``hvp`` of a tangent transported through the ambient space, and the
        quadratic forms, against the ragged problem's (manifold; K = (2,))."""
        rng = np.random.default_rng(29)
        np.random.seed(30)
        struct = ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
        x = t3.TuckerTensorTrain(*_unit_point(struct))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        ww, pp, _ = _pd_sample(x.shape, 9, 2, 1, rng)
        data = [z + 0.1 * rng.standard_normal(z.shape) for z in sd.t3_probe_derivatives(ww, pp, _unit_point(struct), 2)]
        weight = [1.0, 0.5, 0.25]
        rprob = bopt.least_squares_problem(_MANIFOLD, bfit.ProbeDerivativesKind(order=2, weight=weight), (ww, pp), data)
        uprob = uf.uniform_least_squares_problem('manifold', 'probe_derivatives', ux, (ww, pp), data, order=2,
                                                 weight=weight)
        rlm = rprob.local_model(x.data, hessian='exact')
        ulm = uprob.local_model((ux.data[0], ux.data[1]), hessian='exact')
        p = rprob.geom.project(rlm.frame, fvo.fv_variations_randn(fvo.fv_variation_shapes(rlm.frame), (2,)))
        p_u = []
        for k in range(2):
            pk = tuple(tuple(a[k] for a in fam) for fam in p)
            proj = utvo.utv_project_ut3_onto_tangent_space(
                ulm.frame, ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*tvo.tv_to_t3(rlm.frame, pk))).data)
            p_u.append(proj[:2])
        p_u = tuple(np.stack([q[j] for q in p_u], axis=1) for j in range(2))
        vm = ufm.ufv_variation_masks_over_stack(ulm.frame[5], (), (2,))
        Hr, Hu = rlm.hvp(p), ulm.hvp(p_u)
        for k in range(2):
            Dr = tvo.tv_to_dense(rlm.frame, tuple(tuple(a[k] for a in fam) for fam in Hr))
            Du = np.asarray(ut3c.ut3_to_dense(utvo.utv_to_ut3(
                ulm.frame, (Hu[0][:, k], Hu[1][:, k], tuple(ulm.frame[4]), tuple(m[:, k] for m in vm)))))
            self.assertLess(np.linalg.norm(Du - Dr) / np.linalg.norm(Dr), 1e-12)
        for act in (lambda m, q: m.quadratic(q), lambda m, q: m.curvature_quadratic(q)):
            a, b = np.asarray(act(rlm, p)), np.asarray(act(ulm, p_u))
            self.assertLess(float(np.max(np.abs(a - b) / np.abs(a))), 1e-12)

    def test_flat_slots(self):
        """T20 on the uniform layer: at ``d = 1`` the manifold model carries no curvature data and its hvp IS the
        Gauss-Newton one (the shortcut); at maximal rank the curvature term vanishes against the sampling term
        it would otherwise be (no special case: the normal part of the residual is zero there)."""
        rng = np.random.default_rng(33)
        np.random.seed(34)
        for struct in (((5,), (1,), (1, 1)), ((2, 3, 2), (2, 3, 2), (1, 2, 2, 1))):   # (uniform: minimal ranks)
            with self.subTest(struct=struct):
                x = t3.TuckerTensorTrain(*_unit_point(struct))
                ux = ut3.UniformTuckerTensorTrain.from_t3(x)
                ww, pp, _ = _pd_sample(x.shape, 8, 1, 1, rng)
                data = [z + rng.standard_normal(z.shape) for z in sd.t3_probe_derivatives(ww, pp, x.data, 1)]
                prob = uf.uniform_least_squares_problem('manifold', 'probe_derivatives', ux, (ww, pp), data, order=1)
                sc = (ux.data[0], ux.data[1])
                ex, gn = prob.local_model(sc, hessian='exact'), prob.local_model(sc)
                p = prob.geom.project(ex.frame, _random_variation_sc(ex.frame, (), rng))
                if len(struct[0]) == 1:
                    self.assertIsNone(ex.curvature.geometry_data)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(ex.hvp(p), gn.hvp(p))))
                    continue
                cd = ex.curvature.geometry_data
                theta_hat = prob.geom.lift(ex.frame, p, cd)
                K_r = ex.kind.corewise_hessian(theta_hat, ex.residual, ex.sample, ex.frame, ex.sweep,
                                               prob.geom.base_point(ex.frame))
                LT_Kr = prob.geom.lift_transpose(ex.frame, K_r, cd)
                nrm = lambda t: float(np.sqrt(sum(np.sum(np.asarray(a) ** 2) for a in t)))
                self.assertGreater(nrm(LT_Kr), 1e-3)
                self.assertLess(nrm(ex.curvature_hvp(p)), 1e-12 * nrm(LT_Kr))


class TestUniformKStackedGaussNewton(unittest.TestCase):
    """The S0c m1 gap: a uniform kind's ``forward`` (and so the Gauss-Newton ``hvp`` and the geometry's
    ``project`` / ``inner``) takes a ``K``-stacked variation at a ``C`` frame, per element equal to the ragged."""

    def test_forward_and_gn_hvp_on_a_tangent_stack(self):
        rng = np.random.default_rng(35)
        for name in ('d3 varying', 'd4 varying'):
            with self.subTest(case=name):
                lm, rlm, uframe, vshapes, n = _exact_models(name, 36, rng)
                gn = dc.replace(lm, curvature=None)
                rgn = [dc.replace(m, curvature=None) for m in rlm]
                K = (3,)
                p = _random_variation_sc(uframe, K, rng)
                p_r = [tuple(_ragged_slices(p[j], K, n, lambda cc, j=j: vshapes(cc)[j])[c] for j in range(2))
                       for c in range(n)]
                z = gn.kind.forward(p, gn.sample, gn.frame, gn.sweep)               # (d,)+(o+1,)+W+K+C+(N,)
                for c in range(n):
                    zr = rgn[c].kind.forward(p_r[c], rgn[c].sample, rgn[c].frame, rgn[c].sweep)
                    for i, zi in enumerate(zr):
                        zu = np.asarray(z)[i][..., c, :zi.shape[-1]] if n > 1 else np.asarray(z)[i][..., :zi.shape[-1]]
                        self.assertLess(np.linalg.norm(zu - zi) / np.linalg.norm(zi), _TOL)
                Hp = gn.hvp(p)
                refs = [rgn[c].hvp(p_r[c]) for c in range(n)]
                for j in range(2):
                    _compare(self, Hp[j], lambda c, j=j: refs[c][j], K, n, 'gn hvp %d' % j)
                self.assertEqual(np.shape(gn.geom.inner(p, p)), K + ((n,) if n > 1 else ()))


# ---------------------------------------------------------------------------------------------------
# End to end: exact newton_cg on the uniform layer (T21, T25)
# ---------------------------------------------------------------------------------------------------
def _same_run(test, a, b, tol=1e-8):
    """The uniform run is the ragged run, to the equivalence contract: the same outer / accepted / CG counts and
    stop, and the objective row by row."""
    test.assertEqual(a['stop_reason'], b['stop_reason'])
    test.assertEqual((a['newton_steps'], a['newton_accepted']), (b['newton_steps'], b['newton_accepted']))
    test.assertEqual([row['cg_iters'] for row in a['history']], [row['cg_iters'] for row in b['history']])
    test.assertEqual(len(a['losses']), len(b['losses']))
    worst = max(abs(x - y) / abs(y) for x, y in zip(a['losses'], b['losses']))
    test.assertLess(worst, tol)
    return worst


def _restart_template(W, level1_max_newton=80):
    """T21's template (c24 on probe derivatives; tests/backend/test_curvature.py): order 2, omega = [1, .5, .25],
    5 % noise, target rank 4, the level-1 rank-2 fit zero-padded to rank 3. ``level1_max_newton`` < 80 truncates
    the level-1 fit (the always-on variant's cheaper template: still a zero-padded, rank-deficient restart)."""
    rng = np.random.default_rng(7)
    np.random.seed(7)
    shape = (6, 6, 6)
    target = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    pp = [rng.standard_normal((W, N)) for N in shape]
    weight = np.array([1.0, 0.5, 0.25])
    kind = bfit.probe_derivatives_kind(2, weight)
    data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(target.data, (ww, pp))]
    problem = bopt.least_squares_problem(_MANIFOLD, kind, (ww, pp), data)
    x1, _ = bopt.newton_cg(problem, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data,
                           max_newton=level1_max_newton, gtol_rel=1e-10)
    restart = t3.TuckerTensorTrain(*x1).resize(shape, (3, 3, 3), (1, 3, 3, 1))
    return problem, (ww, pp), data, weight, restart


_SLOW = unittest.skipUnless(os.environ.get('T3TOOLBOX_SLOW_TESTS'),
                            'the full-length run (> 3 s): set T3TOOLBOX_SLOW_TESTS=1 (the release gate)')


class TestUniformExactNewton(unittest.TestCase):
    """T21 / T25 on the uniform layer, to the equivalence contract (``docs/uniform_equivalence_contract.md``
    §"The full-rank qualifier"): from a numerically FULL-RANK start -- the slack starts, reduced to their minimal
    ranks by the frontend, and the same minimal tensor through the uniform backend -- exact ``newton_cg`` runs the
    RAGGED run: the same outer / accepted / CG counts and stop, the objective row by row to 1e-8. From the
    zero-padded restart (T21) the start is rank-DEFICIENT, where the two layers are required to choose different
    null-space completions of the frame (so different tangent spaces and trajectories, by design); there the
    uniform run must meet T21 itself -- converge on the gradient test with at most two rejections from a singular
    lift -- and reach the ragged run's minimizer; and the frontend uniform run IS the backend one. Observed (S8):
    T21 ragged and uniform both 8 outer / 0 rejected / 27 CG (per-step CG 1,1,2,2,3,4,6,8 vs 1,1,1,2,3,4,6,9: the
    first gradients differ by 8 %, the completions), the same final objective to the last bit; T25 identical counts
    (14 / 1 / 26 and 22 / 6 / 70), objectives row by row to <= 1.9e-13.

    **Budget** (plan §F: <= 3 s each, or the slow tier): the always-on T21 truncates the level-1 fit of its
    template (12 GN steps -- still a zero-padded, rank-deficient restart; ~2.1 s) and the always-on TT-slack T25
    runs the first 10 Newton rows (2 of them rejected; ~1.3 s); their full-length versions -- T21 on the ragged
    T21's own template (3.6 s) and the TT-slack run to convergence (4.5 s) -- are in the slow tier
    (``T3TOOLBOX_SLOW_TESTS=1``, a required release-gate step)."""

    def test_restart_backend_and_frontend(self):
        self._restart(level1_max_newton=12)             # observed: ragged 8 / 0, uniform 9 / 0 (rejected) outer

    @_SLOW
    def test_restart_backend_and_frontend_full_template(self):
        self._restart(level1_max_newton=80)             # the ragged T21's template; observed 8 / 0 / 27 both

    def _restart(self, level1_max_newton):
        import t3toolbox.optimizers as topt
        import t3toolbox.uniform_manifold as ut3m
        problem, sample, data, weight, restart = _restart_template(60, level1_max_newton)
        _, ragged = bopt.newton_cg(problem, restart.data, max_newton=40, hessian='exact')
        self.assertEqual(ragged['stop_reason'], 'gradient')
        ux = ut3.UniformTuckerTensorTrain.from_t3(restart)
        uprob = uf.uniform_least_squares_problem('manifold', 'probe_derivatives', ux, sample, data, order=2,
                                                 weight=weight, chunk_size=problem.kind.chunk_size)
        _, ub = bopt.newton_cg(uprob, (ux.data[0], ux.data[1]), max_newton=40, hessian='exact')
        h = ub['history']
        self.assertEqual(ub['stop_reason'], 'gradient')
        self.assertLessEqual(sum(row['tr_accepted'] is False for row in h), 2)
        self.assertEqual(h[0]['lift_sigma_min'], 0.0)                       # the padded slots: a singular lift
        self.assertTrue(all(np.isfinite(row['pKp']) for row in h if not row['converged']))
        self.assertLessEqual(ub['newton_steps'], 12)
        self.assertLess(abs(ub['losses'][-1] - ragged['losses'][-1]) / ragged['losses'][-1], 1e-9)   # same minimizer
        _, uf_ = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', sample, data, ux, order=2, weight=weight,
                                max_newton=40, hessian='exact', chunk_size=problem.kind.chunk_size)
        self.assertEqual(uf_['losses'], ub['losses'])                       # the frontend IS the backend run
        self.assertEqual([row['cg_iters'] for row in uf_['history']], [row['cg_iters'] for row in h])

    def test_tucker_slack_start_backend_and_frontend(self):
        self._slack_start(100, 1, ((4, 4, 4), (1, 2, 2, 1)))         # observed: 14 outer / 1 rejected / 26 CG

    def test_tt_slack_start_backend_and_frontend(self):
        self._slack_start(200, 0, ((4, 4, 4), (1, 9, 9, 1)), max_newton=10)   # the first 10 rows (rows 2, 5 rejected)

    @_SLOW
    def test_tt_slack_start_backend_and_frontend_to_convergence(self):
        self._slack_start(200, 0, ((4, 4, 4), (1, 9, 9, 1)))         # observed: 22 outer / 6 rejected / 70 CG

    def _slack_start(self, W, order, ranks, max_newton=80):
        """T25's set-up (tests/backend/test_curvature.py ``_slack_problem``): the ragged run from the minimal-rank
        tensor (T25: the same run as from the slack start), against the uniform backend from that minimal tensor and
        the uniform FRONTEND from the slack start itself (which reduces it to minimal ranks, on the uniform layer).
        ``max_newton`` < 80 compares the first rows only (every run then stops on ``max_newton``)."""
        import t3toolbox.optimizers as topt
        import t3toolbox.uniform_manifold as ut3m
        rng = np.random.default_rng(11)
        np.random.seed(11)
        shape = (6, 6, 6)
        target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
        ww = [rng.standard_normal((W, N)) for N in shape]
        pp = [rng.standard_normal((W, N)) for N in shape]
        kind = bfit.ProbeDerivativesKind(order=order)
        data = [z + 0.02 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
                for z in kind.point_forward(target.data, (ww, pp))]
        x0 = t3.TuckerTensorTrain.randn(shape, *ranks)
        problem = bopt.least_squares_problem(_MANIFOLD, kind, (ww, pp), data)
        minimal = x0.t3svd()[0].rank_adjustment_sweep('right_to_left')
        _, ragged = bopt.newton_cg(problem, minimal.data, max_newton=max_newton, hessian='exact')
        self.assertEqual(ragged['stop_reason'], 'gradient' if max_newton == 80 else 'max_newton')
        if max_newton < 80:
            self.assertTrue(any(row['tr_accepted'] is False for row in ragged['history']))   # still discriminating
        um = ut3.UniformTuckerTensorTrain.from_t3(minimal)                       # the backend, minimal start
        uprob = uf.uniform_least_squares_problem('manifold', 'probe_derivatives', um, (ww, pp), data,
                                                 order=order, chunk_size=kind.chunk_size)
        _, ub = bopt.newton_cg(uprob, (um.data[0], um.data[1]), max_newton=max_newton, hessian='exact')
        _same_run(self, ub, ragged)
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)                           # the frontend, slack start
        _, uni = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'probe_derivatives', (ww, pp), data, ux0, order=order,
                                max_newton=max_newton, hessian='exact', chunk_size=kind.chunk_size)
        _same_run(self, uni, ragged)

if __name__ == '__main__':
    unittest.main()
