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
geometries' exact members (``d = 1`` -> ``None``; shared -> the tied lift on the companion); the backend exact
``LocalModel`` (hvp / curvature_hvp / quadratic / curvature_quadratic / gradient, ``K`` stacks) per element and,
through ``uniform_least_squares_problem``, against the ragged problem's dense actions; the Gauss-Newton forward and
hvp on a ``K`` stack (the S0c m1 gap); the uniform m27 (the exact gradient IS the Gauss-Newton one); T20 (flat
slots). End to end (T21 / T25): exact ``newton_cg`` from slack starts (reduced to full rank), uniform backend and
frontend row by row equal to the ragged runs; from a zero-padded restart (rank-deficient: the layers' frames
legitimately differ there) each converges to the ragged run's minimizer, and the frontend IS the backend run.
The full-length T21 / TT-slack runs are slow-tier (``T3TOOLBOX_SLOW_TESTS=1``).

S9, the SHARED uniform layer (T12's uniform twin under the T17 rules, T22): the uniform tied lift factors / lift /
transpose and the shared exact model per element vs the ragged shared ones, tied outputs, garbage in the
companion's padding included; the FD and dense-Lagrangian oracles on the uniform Problem path; the zero-padded and
group-ceiling restarts end to end, and the full-rank group-ceiling start row by row equal to the ragged run.
"""
import dataclasses as dc
import os
import sys
import unittest

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.corewise as cw
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.t3_conversions as t3c
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _curvature_oracles as oracles

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


def _compare(test, u, ragged_of, K, n_elements, label, tol=_TOL, scale_of=None):
    """Per element vs ragged, and the exact output mask: ``u`` (a supercore ``(d,)+K+C+trail``) equals
    ``ragged_of(c)`` (a ``K``-stacked tuple of ``d`` ragged arrays) on each element's real blocks -- whose extents
    come from the RAGGED arrays' shapes -- and is exactly zero everywhere else. The error is relative to the
    family's own norm, or to ``scale_of(c, k)`` when given (an output that is a cancelled sum: ``_hvp_term_scale``)."""
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
            if scale_of is not None:
                den = float(scale_of(c, k)) ** 2
            worst = max(worst, (num / max(den, 1e-300)) ** 0.5)
    test.assertTrue(np.all(u[~ind] == 0.0), '%s: a padded slot is not exactly zero' % label)
    test.assertLess(worst, tol, label)
    return worst


def _pair_relerr(a, b):
    """||a - b|| / ||b|| over a whole (Tucker, TT) supercore pair."""
    num = sum(float(np.sum((np.asarray(x) - np.asarray(y)) ** 2)) for x, y in zip(a, b))
    den = sum(float(np.sum(np.asarray(y) ** 2)) for y in b)
    return (num / den) ** 0.5


def _hvp_term_scale(m, p, K, curvature_only=False):
    """The rounding scale of the ragged model ``m``'s exact ``hvp(p)`` (``curvature_only``: ``curvature_hvp(p)``),
    per ``K`` element: the sum of the norms of the terms it adds, each over the whole tangent (Tucker + TT) --
    ``|GN| + |L^T K_a th| + |L^T K_g th|``, ``th = L Pi p``, ``K_a`` the kind's two-hole at the residual, ``K_g`` the
    tangential curvature. The sum cancels: at the CI failure (entries, manifold, 'd3 varying', K = (2,); element 0,
    k = 0) the TT block of ``H p`` is 26 against TT terms of 208 and 198 (raw ``K_a th``, ``K_g th`` 144 each) and a
    Tucker block of 261, so a TT error set by those terms is ~25x its own norm; relative to it, a BLAS-order change
    moved the error 3.9e-14 -> 1.2e-13 (numpy 2.5 -> 2.4). Against this scale that subtest is 7e-15 and every one
    is <= 1.3e-14 (numpy 1.26) / 8.5e-15 (2.x): the rest is the lift's conditioning (|th| / |Pi p| ~ 125 there)."""
    geom, data = m.geom, m.curvature.geometry_data
    pp = geom.project(m.frame, p, aux=m.geom_aux)
    terms = [] if curvature_only else [
        m.kind.transpose(m.kind.forward(pp, m.sample, m.frame, m.sweep), m.sample, m.frame, m.sweep)]
    if data is not None:
        th = geom.lift(m.frame, pp, data)
        k_g = geom.tangential_curvature(m.frame, th, data)
        terms += [geom.lift_transpose(m.frame, kt, data) for kt in (
            m.kind.corewise_hessian(th, m.residual, m.sample, m.frame, m.sweep, geom.base_point(m.frame)), k_g)
            if kt is not None]
    knorm = lambda t: np.sqrt(sum(np.sum(np.asarray(a) ** 2, axis=tuple(range(len(K), np.ndim(a))))
                                  for a in t[0] + t[1]))
    return sum((knorm(t) for t in terms), np.zeros(K))


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
    """``UniformProbeDerivativesKind.corewise_hessian``: its own masked override (T23); its masks are
    the FRAME's ``(up, left)`` (m29: at TT slack ``frame[5]`` is a strict subset of ``self.masks``, and the
    point masks would let the frame's padded completions into theta); T24: theta masked == the sweep's
    ``(mb[0], mb[2])``, and garbage in theta_hat's padded slots (every edge) == zeroed padding, the real block
    bitwise."""

    def test_the_override_is_its_own(self):
        self.assertIn('corewise_hessian', vars(uf.UniformProbeDerivativesKind))          # its own masked override
        self.assertIsNot(uf.UniformProbeDerivativesKind.corewise_hessian, bfit.ProbeDerivativesKind.corewise_hessian)
        self.assertFalse(getattr(uf.UniformProbeDerivativesKind.corewise_hessian, '__isabstractmethod__', False))
        mro = uf.UniformProbeDerivativesKind.__mro__
        self.assertLess(mro.index(uf._UniformLayerMixin), mro.index(bfit.ProbeDerivativesKind))   # precedes

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
# The plain kinds' uniform two-holes (S10 / S11): probe, apply, entries
# ---------------------------------------------------------------------------------------------------
_PLAIN_KINDS = ('probe', 'apply', 'entries')
_PLAIN_UNIFORM = {'probe': (ut3s.ut3_probe_corewise_hessian, ut3s.ut3_probe_corewise_hessian_from_sweep,
                            utvs.utv_precompute_probe_frame_sweep),
                  'apply': (ut3s.ut3_apply_corewise_hessian, ut3s.ut3_apply_corewise_hessian_from_sweep,
                            utvs.utv_precompute_apply_frame_sweep),
                  'entries': (ut3s.ut3_entries_corewise_hessian, ut3s.ut3_entries_corewise_hessian_from_sweep,
                              utvs.utv_precompute_entries_frame_sweep)}


def _plain_ragged_hessian(kind):
    import t3toolbox.backend.apply as bapply
    import t3toolbox.backend.entries as bentries
    import t3toolbox.backend.probing as probing
    return {'probe': probing.t3_probe_corewise_hessian, 'apply': bapply.t3_apply_corewise_hessian,
            'entries': bentries.t3_entries_corewise_hessian}[kind]


def _plain_sample(kind, shape, W, n, rng):
    """A ragged sample (ww, or an index grid for entries) and a residual with the C axis (n > 1) in the kind's own
    layout, plus the per-element residual ``r_of(c)``."""
    C = (n,) if n > 1 else ()
    s = (np.stack([rng.integers(0, N, size=W) for N in shape]) if kind == 'entries'
         else [rng.standard_normal((W, N)) for N in shape])
    if kind == 'probe':
        z = [rng.standard_normal((W,) + C + (N,)) for N in shape]
        return s, z, (lambda c: [zi[..., c, :] for zi in z] if n > 1 else z)
    z = rng.standard_normal((W,) + C)
    return s, z, (lambda c: z[..., c] if n > 1 else z)


class TestUniformPlainKindsTwoHole(unittest.TestCase):
    """``ut3_{probe,apply,entries}_corewise_hessian[_from_sweep]`` (T17): per element the ragged two-hole (``K``
    stacks, ``d`` = 1, 3, 4, varying ranks, TT slack), exact output masks, and garbage in the padding of theta,
    theta_hat, the residual and the vectors ignored (bitwise); the sweep-reusing form equals the one-shot form; and the
    direction mask is load-bearing (the theta_hat-mask-removed mutant corrupts the real block)."""

    def test_per_element_vs_ragged(self):
        rng = np.random.default_rng(91)
        W = 7
        for kind in _PLAIN_KINDS:
            hess, from_sweep, precompute = _PLAIN_UNIFORM[kind]
            ragged_hess = _plain_ragged_hessian(kind)
            for name in _CASES:
                for K in _KS:
                    with self.subTest(kind=kind, case=name, K=K):
                        _points, rframes, uframe, _dims, n = _case(name, 92)
                        shape = tuple(uframe[4])
                        s, z, z_of = _plain_sample(kind, shape, W, n, rng)
                        theta = _theta_data(uframe)
                        D = _random_theta_sc(uframe, K, rng)
                        D_r = [tuple(_ragged_slices(D[j], K, n, lambda cc, j=j: _theta_shapes(rframes[cc])[j])[c]
                                     for j in range(2)) for c in range(n)]
                        out = hess(z, s, theta, D)
                        ref = [ragged_hess(z_of(c), s, (rframes[c][0], rframes[c][2]), D_r[c]) for c in range(n)]
                        for j in range(2):
                            _compare(self, out[j], lambda c, j=j: ref[c][j], K, n, 'two-hole %d' % j)
                        # garbage everywhere padded: theta (the frame), the residual and the vectors (beyond N_i)
                        N = uframe[0].shape[-1]
                        dirty_theta = _dirty_ut3(theta[:2], theta[2], theta[3], seed=3) + theta[2:]
                        pk = lambda v, sd_: _dirty_packed_vectors(ut3o.pack_if_ragged(v, N), shape, sd_)
                        dz = pk(z, 4) if kind == 'probe' else z
                        ds = s if kind == 'entries' else pk(s, 5)
                        again = hess(dz, ds, dirty_theta, D)
                        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(out, again)), 'garbage leaked')
                        sweep = precompute(ufc.ut3_corewise_frame(theta), s)
                        reuse = from_sweep(z, theta, D, sweep)
                        self.assertLess(_pair_relerr(reuse, out), _TOL)

    def test_the_direction_mask_is_load_bearing(self):
        """Negative control, per kind (F15 / F18): the kernel on the masked theta but the UNmasked direction corrupts
        the real block -- what the wrapper's second mask prevents (even re-masked, the real block is off)."""
        rng = np.random.default_rng(93)
        for kind in _PLAIN_KINDS:
            with self.subTest(kind=kind):
                _points, rframes, uframe, _dims, n = _case('d3', 94)
                shape = tuple(uframe[4])
                s, z, _ = _plain_sample(kind, shape, 6, n, rng)
                theta = _theta_data(uframe)
                D = _random_theta_sc(uframe, (), rng)
                good = _PLAIN_UNIFORM[kind][0](z, s, theta, D)
                N = uframe[0].shape[-1]
                pk = lambda v: ut3o.pack_if_ragged(v, N)
                bare = _plain_ragged_hessian(kind)(pk(z) if kind == 'probe' else z, s if kind == 'entries' else pk(s),
                                                   ut3mask.ut3_apply_masks(theta), D)
                bare = ut3mask.ut3_apply_masks(bare + theta[2:])
                self.assertGreater(np.linalg.norm(bare[1] - good[1]), 1e-3 * np.linalg.norm(good[1]))


class TestUniformPlainKindCorewiseHessian(unittest.TestCase):
    """``UniformProbeKind`` / ``UniformApplyKind`` / ``UniformEntriesKind.corewise_hessian`` (T23 after the override,
    T24, m29), in the S8 shape: each is its own masked override; its masks are the FRAME's ``(up, left)`` -- at
    TT slack ``frame[5]`` is a strict subset of ``self.masks``, and the point masks give a different answer (the
    ``self.masks``-instead-of-frame mutant); garbage in theta_hat's padded slots == zeroed padding, bitwise."""

    _UKIND = {'probe': lambda d: uf.UniformProbeKind.from_point(d, weight=[0.5, 1.0, 2.0]),
              'apply': lambda d: uf.UniformApplyKind.from_point(d),
              'entries': lambda d: uf.UniformEntriesKind.from_point(d)}
    _RKIND = {'probe': lambda: bfit.ProbeKind(weight=[0.5, 1.0, 2.0]), 'apply': lambda: bfit.APPLY,
              'entries': lambda: bfit.ENTRIES}

    _CLASSES = ((uf.UniformProbeKind, bfit.ProbeKind), (uf.UniformApplyKind, bfit.ApplyKind),
                (uf.UniformEntriesKind, bfit.EntriesKind))

    def test_the_override_is_its_own(self):
        for cls, ragged in self._CLASSES:
            with self.subTest(kind=cls.__name__):
                self.assertIn('corewise_hessian', vars(cls))                         # its own masked override
                self.assertIsNot(cls.corewise_hessian, ragged.corewise_hessian)      # never the mask-free ragged one
                self.assertFalse(getattr(cls.corewise_hessian, '__isabstractmethod__', False))
                mro = cls.__mro__
                self.assertLess(mro.index(uf._UniformLayerMixin), mro.index(ragged))      # precedes

    def test_masks_from_the_frame_at_tt_slack(self):
        rng = np.random.default_rng(95)
        x, ux, rframe, frame = _tt_slack_uniform(96)
        point_up, point_left = ux.data[3]
        self.assertTrue(np.all(frame[5][2] <= point_left) and not np.array_equal(frame[5][2], point_left))   # F23
        shape, W = x.shape, 8
        base = (frame[0], frame[2])
        for kind in _PLAIN_KINDS:
            ukind, rkind = self._UKIND[kind](ux.data), self._RKIND[kind]()
            s, z, _ = _plain_sample(kind, shape, W, 1, rng)
            sample = uf.pack_sample(kind, s, ux.N)
            sweep = ukind.precompute(frame, sample)
            zp = uf.pack_data(kind, z, ux.N)
            rsweep = rkind.precompute(rframe, s)
            for K in _KS:
                with self.subTest(kind=kind, K=K):
                    D = _random_theta_sc(frame, K, rng)
                    out = ukind.corewise_hessian(D, zp, sample, frame, sweep, base)
                    D_r = tuple(_ragged_slices(D[j], K, 1, lambda c, j=j: _theta_shapes(rframe)[j])[0] for j in range(2))
                    ref = rkind.corewise_hessian(D_r, z, s, rframe, rsweep, (rframe[0], rframe[2]))
                    for j in range(2):                     # the extent is the FRAME's (the ragged frame's shapes)
                        _compare(self, out[j], lambda c, j=j: ref[j], K, 1, 'kind %d' % j)
                    wrong = _PLAIN_UNIFORM[kind][1](                                    # the point's masks instead
                        ukind._apply_weight(zp, 2), (base[0], base[1], ux.data[2], ux.data[3]), D, sweep)
                    self.assertFalse(np.allclose(wrong[1], out[1]), 'the point masks gave the same answer')

    def test_theta_hat_masking_unit(self):
        """T24: garbage in theta_hat's padded slots (every edge padded) gives the zeroed-padding answer, bitwise."""
        rng = np.random.default_rng(97)
        _points, rframes, uframe, _dims, n = _case('d3 varying', 98)
        shape, W = tuple(uframe[4]), 5
        masks = (uframe[5][0], uframe[5][2])
        for m in masks:                                                  # every edge has padding
            self.assertTrue(np.all(~m[..., -1]))
        for kind in _PLAIN_KINDS:
            with self.subTest(kind=kind):
                ukind = self._UKIND[kind]((None, None, shape, masks))
                s, z, _ = _plain_sample(kind, shape, W, n, rng)
                sample = uf.pack_sample(kind, s, uframe[0].shape[-1])
                zp = uf.pack_data(kind, z, uframe[0].shape[-1])
                sweep = ukind.precompute(uframe, sample)
                D = _random_theta_sc(uframe, (2,), rng)
                clean = ut3mask.ut3_apply_masks(D + (shape, masks))
                a = ukind.corewise_hessian(D, zp, sample, uframe, sweep, (uframe[0], uframe[2]))
                b = ukind.corewise_hessian(tuple(clean), zp, sample, uframe, sweep, (uframe[0], uframe[2]))
                self.assertTrue(all(np.array_equal(p, q) for p, q in zip(a, b)))
                self.assertGreater(min(np.abs(np.asarray(D[j]) - np.asarray(clean[j])).max() for j in range(2)), 1.0)


# ---------------------------------------------------------------------------------------------------
# The apply / entries derivative kinds' uniform two-holes (S12)
# ---------------------------------------------------------------------------------------------------
_DERIV_KINDS = ('apply_derivatives', 'entries_derivatives')


def _deriv_uniform(kind, order):
    """(the uniform two-hole, its from-sweep form, the uniform jet sweep builder, the ragged two-hole) for a derivative
    kind at ``order``, in one call shape: ``s`` is the paired sample ``(ww, pp)`` / ``(index, pp)``."""
    if kind == 'apply_derivatives':
        return (lambda c, s, th, dd: ut3s.ut3_apply_corewise_derivatives_hessian(c, s[0], s[1], th, dd, order),
                lambda c, th, dd, sw: ut3s.ut3_apply_corewise_derivatives_hessian_from_sweep(c, th, dd, order, sw),
                lambda f, s: utvs.utv_precompute_apply_frame_sweep_jets(f, s[0], s[1], order),
                lambda c, s, th, dd: sd.t3_apply_corewise_derivatives_hessian(c, s[0], s[1], th, dd, order))
    return (lambda c, s, th, dd: ut3s.ut3_entries_corewise_derivatives_hessian(c, s[0], s[1], th, dd, order),
            lambda c, th, dd, sw: ut3s.ut3_entries_corewise_derivatives_hessian_from_sweep(c, th, dd, order, sw),
            lambda f, s: utvs.utv_precompute_entries_frame_sweep_jets(f, s[0], s[1], order),
            lambda c, s, th, dd: sd.t3_entries_corewise_derivatives_hessian(c, s[0], s[1], th, dd, order))


def _deriv_uniform_sample(kind, shape, W, order, n, rng):
    """A ragged paired sample, a residual jet ``(order+1, W)+C`` (``C = (n,)`` when ``n > 1``) and the per-element
    residual ``c_of(c)``."""
    C = (n,) if n > 1 else ()
    first = (np.stack([rng.integers(0, N, size=W) for N in shape]) if kind == 'entries_derivatives'
             else [rng.standard_normal((W, N)) for N in shape])
    pp = [rng.standard_normal((W, N)) for N in shape]
    c = rng.standard_normal((order + 1, W) + C)
    return (first, pp), c, (lambda cc: c[..., cc] if n > 1 else c)


def _dirty_deriv_sample(kind, s, shape, N, seed):
    """The paired sample packed to ``N`` with garbage beyond each ``N_i`` (entries: the index is never packed)."""
    pk = lambda v, sd_: _dirty_packed_vectors(ut3o.pack_if_ragged(v, N), shape, sd_)
    return (s[0] if kind == 'entries_derivatives' else pk(s[0], seed), pk(s[1], seed + 1))


class TestUniformDerivativeKindsTwoHole(unittest.TestCase):
    """``ut3_{apply,entries}_corewise_derivatives_hessian[_from_sweep]`` (T17): per element the ragged two-hole (orders
    0 and 2, ``K`` stacks, ``d`` = 1, 3, 4, varying ranks, TT slack), exact output masks, and garbage in the padding of
    theta, theta_hat and the vectors ignored (bitwise); the sweep-reusing form equals the one-shot form; and the
    direction mask is load-bearing (the theta_hat-mask-removed mutant corrupts the real block)."""

    def test_per_element_vs_ragged(self):
        rng = np.random.default_rng(111)
        W = 7
        for kind in _DERIV_KINDS:
            for order in (0, 2):
                hess, from_sweep, precompute, ragged_hess = _deriv_uniform(kind, order)
                for name in _CASES:
                    for K in _KS:
                        with self.subTest(kind=kind, order=order, case=name, K=K):
                            _points, rframes, uframe, _dims, n = _case(name, 112)
                            shape = tuple(uframe[4])
                            s, c, c_of = _deriv_uniform_sample(kind, shape, W, order, n, rng)
                            theta = _theta_data(uframe)
                            D = _random_theta_sc(uframe, K, rng)
                            D_r = [tuple(_ragged_slices(D[j], K, n, lambda cc, j=j: _theta_shapes(rframes[cc])[j])[c_]
                                         for j in range(2)) for c_ in range(n)]
                            out = hess(c, s, theta, D)
                            ref = [ragged_hess(c_of(c_), s, (rframes[c_][0], rframes[c_][2]), D_r[c_])
                                   for c_ in range(n)]
                            for j in range(2):
                                _compare(self, out[j], lambda c_, j=j: ref[c_][j], K, n, 'two-hole %d' % j)
                            # garbage everywhere padded: theta (the frame) and the vectors (beyond N_i)
                            dirty_theta = _dirty_ut3(theta[:2], theta[2], theta[3], seed=3) + theta[2:]
                            again = hess(c, _dirty_deriv_sample(kind, s, shape, uframe[0].shape[-1], 5), dirty_theta, D)
                            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(out, again)), 'garbage leaked')
                            reuse = from_sweep(c, theta, D, precompute(ufc.ut3_corewise_frame(theta), s))
                            self.assertLess(_pair_relerr(reuse, out), _TOL)

    def test_the_direction_mask_is_load_bearing(self):
        """Negative control, per kind (F15 / F18): the kernel on the masked theta but the UNmasked direction corrupts
        the real block -- what the wrapper's second mask prevents (even re-masked, the real block is off)."""
        rng = np.random.default_rng(113)
        for kind in _DERIV_KINDS:
            with self.subTest(kind=kind):
                hess, _, _, ragged_hess = _deriv_uniform(kind, 2)
                _points, rframes, uframe, _dims, n = _case('d3', 114)
                shape = tuple(uframe[4])
                s, c, _ = _deriv_uniform_sample(kind, shape, 6, 2, n, rng)
                theta = _theta_data(uframe)
                D = _random_theta_sc(uframe, (), rng)
                good = hess(c, s, theta, D)
                N = uframe[0].shape[-1]
                packed = (s[0] if kind == 'entries_derivatives' else ut3o.pack_if_ragged(s[0], N),
                          ut3o.pack_if_ragged(s[1], N))
                bare = ragged_hess(c, packed, ut3mask.ut3_apply_masks(theta), D)
                bare = ut3mask.ut3_apply_masks(bare + theta[2:])
                self.assertGreater(np.linalg.norm(bare[1] - good[1]), 1e-3 * np.linalg.norm(good[1]))


class TestUniformDerivativeKindCorewiseHessian(unittest.TestCase):
    """``UniformApplyDerivativesKind`` / ``UniformEntriesDerivativesKind.corewise_hessian`` (T23, T24, m29), in the S8
    shape: each is its own masked override; its masks are the FRAME's ``(up, left)`` -- at TT slack ``frame[5]`` is a
    strict subset of ``self.masks``, and the point masks give a different answer (the ``self.masks``-instead-of-frame
    mutant); the per-order weight enters as ``omega^2 r``; garbage in theta_hat's padded slots == zeroed padding,
    bitwise."""

    _ORDER, _WEIGHT = 1, [1.0, 0.5]
    _UCLS = {'apply_derivatives': uf.UniformApplyDerivativesKind,
             'entries_derivatives': uf.UniformEntriesDerivativesKind}
    _RCLS = {'apply_derivatives': bfit.ApplyDerivativesKind, 'entries_derivatives': bfit.EntriesDerivativesKind}

    def test_the_override_is_its_own(self):
        for kind in _DERIV_KINDS:
            cls, ragged = self._UCLS[kind], self._RCLS[kind]
            with self.subTest(kind=cls.__name__):
                self.assertIn('corewise_hessian', vars(cls))                         # its own masked override
                self.assertIsNot(cls.corewise_hessian, ragged.corewise_hessian)      # never the mask-free ragged one
                self.assertFalse(getattr(cls.corewise_hessian, '__isabstractmethod__', False))
                mro = cls.__mro__
                self.assertLess(mro.index(uf._UniformLayerMixin), mro.index(ragged))

    def test_masks_from_the_frame_at_tt_slack(self):
        rng = np.random.default_rng(115)
        x, ux, rframe, frame = _tt_slack_uniform(116)
        point_up, point_left = ux.data[3]
        self.assertTrue(np.all(frame[5][2] <= point_left) and not np.array_equal(frame[5][2], point_left))   # F23
        shape, W, order = x.shape, 8, self._ORDER
        base = (frame[0], frame[2])
        for kind in _DERIV_KINDS:
            ukind = self._UCLS[kind].from_point(ux.data, order=order, weight=self._WEIGHT)
            rkind = self._RCLS[kind](order=order, weight=self._WEIGHT)
            s, c, _ = _deriv_uniform_sample(kind, shape, W, order, 1, rng)
            sample = uf.pack_sample(kind, s, ux.N)
            sweep = ukind.precompute(frame, sample)
            rsweep = rkind.precompute(rframe, s)
            for K in _KS:
                with self.subTest(kind=kind, K=K):
                    D = _random_theta_sc(frame, K, rng)
                    out = ukind.corewise_hessian(D, c, sample, frame, sweep, base)
                    D_r = tuple(_ragged_slices(D[j], K, 1, lambda c_, j=j: _theta_shapes(rframe)[j])[0] for j in range(2))
                    ref = rkind.corewise_hessian(D_r, c, s, rframe, rsweep, (rframe[0], rframe[2]))
                    for j in range(2):                     # the extent is the FRAME's (the ragged frame's shapes)
                        _compare(self, out[j], lambda c_, j=j: ref[j], K, 1, 'kind %d' % j)
                    from_sweep = _deriv_uniform(kind, order)[1]
                    weighted = ukind._apply_weight(c, 2)
                    right = from_sweep(weighted, _theta_data(frame), D, sweep)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(right, out)))         # omega^2 r, frame masks
                    wrong = from_sweep(weighted, (base[0], base[1], ux.data[2], ux.data[3]), D, sweep)   # point masks
                    self.assertFalse(np.allclose(wrong[1], out[1]), 'the point masks gave the same answer')

    def test_theta_hat_masking_unit(self):
        """T24: garbage in theta_hat's padded slots (every edge padded) gives the zeroed-padding answer, bitwise."""
        rng = np.random.default_rng(117)
        _points, rframes, uframe, _dims, n = _case('d3 varying', 118)
        shape, W, order = tuple(uframe[4]), 5, self._ORDER
        masks = (uframe[5][0], uframe[5][2])
        for m in masks:                                                  # every edge has padding
            self.assertTrue(np.all(~m[..., -1]))
        for kind in _DERIV_KINDS:
            with self.subTest(kind=kind):
                ukind = self._UCLS[kind].from_point((None, None, shape, masks), order=order, weight=self._WEIGHT)
                s, c, _ = _deriv_uniform_sample(kind, shape, W, order, n, rng)
                sample = uf.pack_sample(kind, s, uframe[0].shape[-1])
                sweep = ukind.precompute(uframe, sample)
                D = _random_theta_sc(uframe, (2,), rng)
                clean = ut3mask.ut3_apply_masks(D + (shape, masks))
                a = ukind.corewise_hessian(D, c, sample, uframe, sweep, (uframe[0], uframe[2]))
                b = ukind.corewise_hessian(tuple(clean), c, sample, uframe, sweep, (uframe[0], uframe[2]))
                self.assertTrue(all(np.array_equal(p, q) for p, q in zip(a, b)))
                self.assertGreater(min(np.abs(np.asarray(D[j]) - np.asarray(clean[j])).max() for j in range(2)), 1.0)


# ---------------------------------------------------------------------------------------------------
# The uniform geometries' exact members and the exact LocalModel
# ---------------------------------------------------------------------------------------------------
def _uniform_exact_model(uframe, plain_masks, kind, sample, residual, curvature_damping=None, geom=None):
    """A backend exact LocalModel AT A GIVEN uniform frame (``Problem.local_model`` builds its own frame, which is
    only gauge-equivalent to the ragged one): so the coordinates match the ragged model's slot for slot. A shared
    geometry's companion is built once, as ``Problem.local_model`` does (``None`` unshared)."""
    geom = geom or bgeo.UniformManifoldGeometryOps.from_frame(uframe, plain_masks)
    aux = geom.precompute(uframe)
    sweep = kind.precompute(uframe, sample)
    mg = geom.project(uframe, kind.transpose(residual, sample, uframe, sweep), aux)
    cd = geom.curvature_precompute(uframe, mg, aux, curvature_damping)
    return bopt.LocalModel(geom, kind, sample, uframe, sweep, residual, kind.w_axes(sample), None, aux,
                           bopt.CurvatureState(mg, cd))


def _ragged_exact_model(rframe, kind, sample, residual, curvature_damping=None, geom=_MANIFOLD):
    aux = geom.precompute(rframe)
    sweep = kind.precompute(rframe, sample)
    mg = geom.project(rframe, kind.transpose(residual, sample, rframe, sweep), aux)
    cd = geom.curvature_precompute(rframe, mg, aux, curvature_damping)
    return bopt.LocalModel(geom, kind, sample, rframe, sweep, residual, kind.w_axes(sample), None, aux,
                           bopt.CurvatureState(mg, cd))


class TestUniformGeometryExactMembers(unittest.TestCase):
    """The uniform geometries' four exact-mode members per element vs the ragged geometry's, on a frame packed
    from the ragged one: the curvature data (``grad f`` as a doubled-rank T3, densely; the real-slot sigma_min;
    eps), the lift, its transpose and the tangential term; ``d = 1`` gives ``None``; a shared uniform geometry
    builds the TIED lift on its companion (S9; the shared classes below test it); the corewise geometry's are the
    identity / ``()`` / ``None`` and reject a damping."""

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
        for base in (bgeo.UniformManifoldGeometryOps.from_point(ux.data),          # S9: shared runs (was S8's NIE)
                     bgeo.UniformCorewiseGeometryOps.from_point(ux.data)):
            with self.subTest(shared=type(base).__name__):
                shared = base.with_sharing((0, 0, 1))
                f = shared.frame((ux.data[0], ux.data[1]))
                aux = shared.precompute(f)
                mg = shared.project(f, _random_variation_sc(f, (), rng), aux)
                data = shared.curvature_precompute(f, mg, aux, None)
                if isinstance(base, bgeo.UniformCorewiseGeometryOps):
                    self.assertEqual(data, ())
                    continue
                self.assertIs(data.lift_factors.shared_data, aux)                     # the TIED lift, on the companion
                self.assertEqual(len(data.lift_factors.tied_filters), 1)
                again = shared.curvature_precompute(f, mg, None, None)                # aux=None: recomputed, equal
                self.assertTrue(np.array_equal(again.lift_factors.tied_filters[0], data.lift_factors.tied_filters[0]))
        cgeo = bgeo.UniformCorewiseGeometryOps.from_point(ux.data)
        f = cgeo.frame((ux.data[0], ux.data[1]))
        self.assertEqual(cgeo.curvature_precompute(f, None, None, None), ())
        with self.assertRaisesRegex(ValueError, 'no effect on a corewise geometry'):
            cgeo.curvature_precompute(f, None, None, 1e-6)
        v = (ux.data[0], ux.data[1])
        self.assertIs(cgeo.lift(f, v, ()), v)
        self.assertIs(cgeo.lift_transpose(f, v, ()), v)
        self.assertIsNone(cgeo.tangential_curvature(f, v, ()))


def _kind_setup(kind, shape, W, n, rng, masks, N, order=1, weight=(1.0, 0.5)):
    """The uniform and ragged kinds of ``kind`` at the frame masks ``masks``, the ragged sample, the packed sample and
    residual (garbage-padded beyond each N_i where they are packed), and the per-element ragged residual ``r_of(c)``.
    ``weight`` is the derivative kinds' per-order row; plain probe takes a per-mode row (its first d entries of
    ``(0.5, ..., 2.0)``); apply / entries are unweighted. The apply / entries derivative residual is a scalar jet
    ``(order+1, W)+C`` (not packed); their vectors are packed garbage-padded (entries: ``pp`` only)."""
    C = (n,) if n > 1 else ()
    if kind == 'probe_derivatives':                  # (the original S8 draw order, so its tests are unchanged)
        ww, pp, _ = _pd_sample(shape, W, order, n, rng)
        r = [rng.standard_normal((order + 1, W) + C + (N_,)) for N_ in shape]
        rp = _dirty_packed_vectors(ut3o.pack_if_ragged(r, N), shape, 31)        # garbage beyond N_i
        sample = (_dirty_packed_vectors(ut3o.pack_if_ragged(ww, N), shape, 32),
                  _dirty_packed_vectors(ut3o.pack_if_ragged(pp, N), shape, 33))
        ukind = uf.UniformProbeDerivativesKind.from_point((None, None, shape, masks), order=order,
                                                          weight=list(weight), chunk_size=None)
        rkind = bfit.ProbeDerivativesKind(order=order, weight=list(weight), chunk_size=None)
        return ukind, rkind, (ww, pp), sample, rp, (lambda c: [z[..., c, :] for z in r] if n > 1 else r)
    if kind in _DERIV_KINDS:                         # S12: the apply / entries derivatives (a per-order weight)
        entries = kind == 'entries_derivatives'
        first = (np.stack([rng.integers(0, N_, size=W) for N_ in shape]) if entries
                 else [rng.standard_normal((W, N_)) for N_ in shape])
        pp = [rng.standard_normal((W, N_)) for N_ in shape]
        r = rng.standard_normal((order + 1, W) + C)                              # a scalar jet: never packed
        sample = (first if entries else _dirty_packed_vectors(ut3o.pack_if_ragged(first, N), shape, 32),
                  _dirty_packed_vectors(ut3o.pack_if_ragged(pp, N), shape, 33))
        ucls, rcls = ((uf.UniformEntriesDerivativesKind, bfit.EntriesDerivativesKind) if entries
                      else (uf.UniformApplyDerivativesKind, bfit.ApplyDerivativesKind))
        return (ucls.from_point((None, None, shape, masks), order=order, weight=list(weight)),
                rcls(order=order, weight=list(weight)), (first, pp), sample, r, (lambda c: r[..., c] if n > 1 else r))
    if kind == 'entries':
        rsample = np.stack([rng.integers(0, N_, size=W) for N_ in shape])
        sample = rsample                                                         # an integer grid: never packed
    else:
        rsample = [rng.standard_normal((W, N_)) for N_ in shape]
        sample = _dirty_packed_vectors(ut3o.pack_if_ragged(rsample, N), shape, 32)
    if kind == 'probe':
        w = list(np.linspace(0.5, 2.0, len(shape)))                              # per mode
        r = [rng.standard_normal((W,) + C + (N_,)) for N_ in shape]
        rp = _dirty_packed_vectors(ut3o.pack_if_ragged(r, N), shape, 31)
        return (uf.UniformProbeKind.from_point((None, None, shape, masks), weight=w), bfit.ProbeKind(weight=w),
                rsample, sample, rp, (lambda c: [z[..., c, :] for z in r] if n > 1 else r))
    r = rng.standard_normal((W,) + C)
    ukind = (uf.UniformApplyKind if kind == 'apply' else uf.UniformEntriesKind).from_point((None, None, shape, masks))
    return ukind, (bfit.APPLY if kind == 'apply' else bfit.ENTRIES), rsample, sample, r, (
        lambda c: r[..., c] if n > 1 else r)


def _exact_models(name, seed, rng, geometry='manifold', order=1, W=7, weight=(1.0, 0.5), kind='probe_derivatives'):
    """(uniform exact model at a frame packed from the ragged ones, the ragged exact models per element,
    the uniform frame, the per-element variation shapes, n_elements): residual and sample garbage-padded."""
    _points, rframes, uframe, dims, n = _case(name, seed)
    shape = tuple(uframe[4])
    N = uframe[0].shape[-1]
    masks = (uframe[5][0], uframe[5][2])
    ukind, rkind, rsample, sample, rp, r_of = _kind_setup(kind, shape, W, n, rng, masks, N, order, weight)
    if geometry == 'manifold':
        lm = _uniform_exact_model(uframe, masks, ukind, sample, rp, 1e-4)
        rlm = [_ragged_exact_model(rframes[c], rkind, rsample, r_of(c), 1e-4) for c in range(n)]
        vshapes = lambda c: _var_shapes(rframes[c])
    else:
        cframe = ufc.ut3_corewise_frame(_theta_data(uframe))
        geom = bgeo.UniformCorewiseGeometryOps(shape, masks)
        lm = _uniform_exact_model(cframe, masks, ukind, sample, rp, None, geom=geom)
        rcf = [fvc.t3_corewise_frame((f[0], f[2])) for f in rframes]
        rlm = [_ragged_exact_model(rcf[c], rkind, rsample, r_of(c), None, geom=_COREWISE) for c in range(n)]
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


class TestUniformPlainKindsExactLocalModel(unittest.TestCase):
    """The backend exact ``LocalModel`` for the plain kinds (S10 / S11) on the uniform layer, at a frame packed from
    the ragged one: every exact action per element equal to the ragged model's on both uniform geometries
    (varying-rank stacks, forced padding, garbage-padded residual / sample / trial tangents, ``K`` stacks), exact
    output masks; garbage in every padded slot of the frame -- on the corewise frame ``(U, G, G, G)`` the point's
    padded boundary-bond slots included -- leaves every action bitwise unchanged; and one SHARED-uniform subcase per
    kind (the tied lift on the companion, per element the ragged shared model, the output tied)."""

    _KINDS = _PLAIN_KINDS

    def _per_element(self, lm, rlm, uframe, vshapes, n, K, rng):
        p = _random_variation_sc(uframe, K, rng)
        p_r = [tuple(_ragged_slices(p[j], K, n, lambda cc, j=j: vshapes(cc)[j])[c] for j in range(2)) for c in range(n)]
        refs_of = {}
        for label, act in (('hvp', lambda m, q: m.hvp(q)), ('curvature_hvp', lambda m, q: m.curvature_hvp(q))):
            out = act(lm, p)
            refs = refs_of[label] = [act(rlm[c], p_r[c]) for c in range(n)]
            # an exact hvp is a cancelled sum, so its error is relative to its terms' scale (_hvp_term_scale)
            scales = [_hvp_term_scale(rlm[c], p_r[c], K, curvature_only=label == 'curvature_hvp') for c in range(n)]
            for j in range(2):
                _compare(self, out[j], lambda c, j=j: refs[c][j], K, n, '%s %d' % (label, j),
                         scale_of=lambda c, k: scales[c][k])
        knorm = lambda t: np.sqrt(sum(np.sum(np.asarray(a) ** 2, axis=tuple(range(len(K), np.ndim(a))))
                                      for a in t[0] + t[1]))                     # per K element
        for label, act, hv in (('quadratic', lambda m, q: m.quadratic(q), 'hvp'),
                               ('curvature_quadratic', lambda m, q: m.curvature_quadratic(q), 'curvature_hvp')):
            got = np.asarray(act(lm, p))
            self.assertEqual(got.shape, tuple(K) + ((n,) if n > 1 else ()))
            for c in range(n):
                want = np.asarray(act(rlm[c], p_r[c]))
                g = got[..., c] if n > 1 else got
                # a quadratic form sums signed terms and can cancel far below its terms' scale (|q| down to ~1e-3
                # |p||Hp| here), so its rounding is bounded by that scale, not by |q|: 1e-13 of |p| |H p|, observed
                # <= 5e-15 (numpy 2.x) / 1.1e-14 (1.26); relative to |q| the same forms reach 1.7e-12 on numpy 1.26
                scale = knorm(p_r[c]) * knorm(refs_of[hv][c])
                self.assertTrue(np.all(np.abs(g - want) <= 1e-13 * scale), label)
        for j in range(2):
            _compare(self, lm.gradient[j], lambda c, j=j: rlm[c].gradient[j], (), n, 'gradient')
        return out

    def test_every_exact_action_per_element(self):
        rng = np.random.default_rng(101)
        for kind in self._KINDS:
            for geometry in ('manifold', 'corewise'):
                for name in ('d3 varying', 'd4 varying', 'd1 varying', 'tt slack'):
                    if name == 'd1 varying' and geometry == 'manifold':
                        continue                                  # d = 1: no curvature data on the manifold (T20)
                    for K in _KS:
                        with self.subTest(kind=kind, geometry=geometry, case=name, K=K):
                            lm, rlm, uframe, vshapes, n = _exact_models(name, 102, rng, geometry, kind=kind)
                            self._per_element(lm, rlm, uframe, vshapes, n, K, rng)

    def test_garbage_in_the_frame_padding(self):
        rng = np.random.default_rng(103)
        for kind in self._KINDS:
            for geometry in ('manifold', 'corewise'):
                with self.subTest(kind=kind, geometry=geometry):
                    lm, _rlm, uframe, _vs, n = _exact_models('d3 varying', 104, rng, geometry, kind=kind)
                    dirty = _dirty_frame(uframe, seed=105)
                    if geometry == 'corewise':                 # (U, G, G, G): ONE garbage-padded G, three times
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

    def test_shared_uniform_subcase(self):
        rng = np.random.default_rng(106)
        for kind in self._KINDS:
            for geometry in ('manifold', 'corewise'):
                with self.subTest(kind=kind, geometry=geometry):
                    lm, rlm, uframe, vshapes, n = _shared_exact_models('ceiling varying', 107, rng, geometry, kind=kind)
                    out = self._per_element(lm, rlm, uframe, vshapes, n, (2,), rng)
                    if geometry == 'manifold':                             # the output is TIED
                        vm = ufm.ufv_variation_masks_over_stack(uframe[5], uframe[0].shape[1:-2],
                                                                (2,) + uframe[0].shape[1:-2])
                        res = sharing.ufv_tied_variations_residual((out[0], out[1], tuple(uframe[4]), vm), lm.geom_aux)
                        self.assertLess(float(np.max(res)), 1e-12)
                        q = _random_variation_sc(uframe, (), rng)                 # the tied lift is load-bearing
                        pm = _uniform_per_mode_twin(lm, 1e-4).curvature_hvp(q)
                        self.assertGreater(_pair_relerr(pm, lm.curvature_hvp(q)), 1e-3)
                    else:                                                  # the per-group mean: one slice
                        for group in lm.geom.groups:
                            for ii in group[1:]:
                                self.assertTrue(np.array_equal(out[0][ii], out[0][group[0]]))


class TestUniformDerivativeKindsExactLocalModel(TestUniformPlainKindsExactLocalModel):
    """S12: the same uniform exact-``LocalModel`` checks for the apply / entries derivative kinds (order 1, a per-order
    weight): every exact action per element equal to the ragged model's on both uniform geometries, exact output
    masks, garbage in every padded slot of the frame (the corewise frame's boundary-bond slots included) bitwise
    ignored, and one SHARED-uniform subcase per kind (the tied lift on the companion, the output tied)."""

    _KINDS = _DERIV_KINDS


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


# ---------------------------------------------------------------------------------------------------
# S9: the exact Hessian on the SHARED uniform layer -- the uniform tied lift (T12's uniform twin, T22)
# ---------------------------------------------------------------------------------------------------
# Per case: the ragged structures of the stack elements, tied by the labels. Every case but 'minimal' is
# group-ceiling in at least one element -- a group's rank exceeds a member's own ceiling, so that member's S_i^T is
# wide and the per-mode lift is NOT the tied one (on 'minimal' every group S_i^T is square and the two agree); the
# two-element stacks differ in rank per element, and padding is forced on every family (_frame_dims).
_SH_CASES = {
    'ceiling varying': ([((6, 6, 4), (4, 4, 2), (1, 2, 2, 1)), ((6, 6, 4), (3, 3, 2), (1, 3, 2, 1))], (0, 0, 1)),
    'two groups varying': ([((5, 6, 5, 6), (3, 2, 3, 2), (1, 2, 3, 2, 1)),
                            ((5, 6, 5, 6), (2, 3, 2, 3), (1, 2, 2, 2, 1))], (0, 1, 0, 1)),
    'one group': ([((5, 5, 5), (3, 3, 3), (1, 2, 2, 1))], (0, 0, 0)),
    'minimal': ([((5, 5, 6), (2, 2, 2), (1, 2, 2, 1))], (0, 0, 1)),
}
_SH_DISCRIMINATING = ('ceiling varying', 'two groups varying', 'one group')


def _shared_unit_point(struct, labels):
    """A random TIED T3 (one factor array per group) scaled to ``||X||_HS = 1``."""
    tk, tt = sharing.t3_tie_tucker_factors(t3.TuckerTensorTrain.randn(*struct).data, labels)
    nrm = float(np.linalg.norm(np.asarray(t3c.t3_to_dense((tk, tt)))))
    return tuple(tk), tuple(tt[:-1]) + (tt[-1] / nrm,)


def _shared_case(name, seed):
    """(ragged tied points, their ragged frames, the packed uniform frame, the canonical groups, n_elements)."""
    np.random.seed(seed)
    structs, labels = _SH_CASES[name]
    points = [_shared_unit_point(s, labels) for s in structs]
    rframes = [fvc.t3_orthogonal_representations(x)[0] for x in points]
    return (points, rframes, _pack_frames(rframes, _frame_dims(rframes)),
            sharing.validate_sharing(labels, structs[0][0]), len(structs))


def _compare_levels(test, u, refs, n, label, tol=_TOL):
    """A uniform group-spectrum-shaped array ``C+(q,)`` against per-element ragged ``(q_c,)`` arrays: equal on the
    first ``q_c`` levels (``q_c`` from the RAGGED array's length, not a uniform mask), exactly 0 beyond them."""
    u = np.asarray(u)
    for c in range(n):
        uc, r = (u[c] if n > 1 else u), np.asarray(refs[c])
        test.assertLess(float(np.max(np.abs(uc[:r.shape[-1]] - r)) / np.max(np.abs(r))), tol, label)
        test.assertTrue(np.all(uc[r.shape[-1]:] == 0.0), '%s: a padded level is not exactly 0' % label)


def _ragged_tied_factors(rframes, groups, cd):
    return [sharing.fv_tied_lift_factors(f, cd, sharing.fv_shared_frame_data(f, groups)) for f in rframes]


def _dirty_companion(sfd, rframes, nD_pad, seed=0, scale=1e3):
    """The uniform companion with finite garbage in every PADDED slot: ``U_M``'s padded rows (beyond each group
    mode's real ``nD``) and padded-level columns, ``s_g``'s padded levels (positive, far above any clip), ``Vt_M``'s
    padded-level rows and padded ``nU`` columns -- the real extents read off the RAGGED frames' shapes."""
    rng = np.random.default_rng(seed)
    n = len(rframes)
    out = ([], [], [])
    for gi, group in enumerate(sharing.nontrivial_groups(sfd.groups)):
        U, s, Vt = (np.array(a, copy=True) for a in (sfd.svd_U[gi], sfd.svd_s[gi], sfd.svd_Vt[gi]))
        rows = np.zeros((n, U.shape[-2]), bool)
        levels = np.zeros((n, s.shape[-1]), bool)
        cols = np.zeros((n, Vt.shape[-1]), bool)
        for c, f in enumerate(rframes):
            nDs = [f[1][ii].shape[-2] for ii in group]
            for jj, nd in enumerate(nDs):
                rows[c, jj * nD_pad: jj * nD_pad + nd] = True
            levels[c, :min(sum(nDs), f[0][group[0]].shape[-2])] = True
            cols[c, :f[0][group[0]].shape[-2]] = True
        if n == 1:
            rows, levels, cols = rows[0], levels[0], cols[0]
        U += scale * (1.0 - (rows[..., :, None] & levels[..., None, :])) * rng.standard_normal(U.shape)
        s += (1.0 - levels) * scale * (1.0 + np.abs(rng.standard_normal(s.shape)))
        Vt += scale * (1.0 - (levels[..., :, None] & cols[..., None, :])) * rng.standard_normal(Vt.shape)
        for acc, a in zip(out, (U, s, Vt)):
            acc.append(a)
    return dc.replace(sfd, svd_U=tuple(out[0]), svd_s=tuple(out[1]), svd_Vt=tuple(out[2]))


class TestUniformSharedTiedLift(unittest.TestCase):
    """T12's uniform twin, the lift (``ufv_tied_lift_factors`` + ``utv_corewise_lift[_transpose]``): per element the
    ragged tied factors / lift / transpose (<= 1e-13) on varying-rank shared stacks with forced padding, the group
    modes' slices one value, exact output masks, the adjoint identity on K and C stacks at dampings 0 to 10, and the
    tied lift identity ``DPhi[L v] == E v``. The uniform companion's padded ``s_g`` levels filter to exactly 0 BY THE
    MASKS, not only by the clip -- finite garbage in every padded slot of the companion leaves the factors, the lift
    and its transpose bitwise unchanged, where the clip-only filter picks it up -- and ``sigma_min`` reads the REAL
    ``s_g`` levels only. A companion of another layout is a structural error."""

    def test_tied_lift_factors_per_element(self):
        for name in _SH_CASES:
            for cd in (0.0, 1e-3):
                with self.subTest(case=name, curvature_damping=cd):
                    _pts, rframes, uframe, groups, n = _shared_case(name, 61)
                    usfd = sharing.ufv_shared_frame_data(uframe, groups)
                    ulf = ufo.ufv_tied_lift_factors(uframe, cd, usfd)
                    plain = ufo.ufv_lift_factors(uframe, cd)
                    for a, b in zip((ulf.tucker_pinvs, ulf.bond_pinvs, ulf.tucker_svals, ulf.bond_svals, ulf.eps),
                                    (plain.tucker_pinvs, plain.bond_pinvs, plain.tucker_svals, plain.bond_svals,
                                     plain.eps)):
                        self.assertTrue(np.array_equal(a, b))                     # the per-mode fields, bit for bit
                    self.assertIs(ulf.shared_data, usfd)
                    rlf = _ragged_tied_factors(rframes, groups, cd)
                    self.assertEqual(len(ulf.tied_filters), len(rlf[0].tied_filters))
                    for gi in range(len(ulf.tied_filters)):
                        _compare_levels(self, ulf.tied_filters[gi], [lf.tied_filters[gi] for lf in rlf], n,
                                        'tied filter %d' % gi)
                    sig = np.asarray(bgeo._uniform_lift_sigma_min(ulf, uframe[5])).reshape(-1)
                    self.assertTrue(np.allclose(sig, [float(bgeo._lift_sigma_min(lf)) for lf in rlf], rtol=_TOL,
                                                atol=0.0), sig)
                    self.assertTrue(np.all(sig > 0.0))                            # never a padded level's 0
                    dirty = _dirty_frame(uframe, seed=62)
                    dlf = ufo.ufv_tied_lift_factors(dirty, cd, sharing.ufv_shared_frame_data(dirty, groups))
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(dlf.tied_filters, ulf.tied_filters)))

    def test_lift_and_transpose_per_element(self):
        rng = np.random.default_rng(63)
        for name in _SH_CASES:
            for K in _KS:
                with self.subTest(case=name, K=K):
                    _pts, rframes, uframe, groups, n = _shared_case(name, 64)
                    ulf = ufo.ufv_tied_lift_factors(uframe, 1e-3, sharing.ufv_shared_frame_data(uframe, groups))
                    rlf = _ragged_tied_factors(rframes, groups, 1e-3)
                    shape = tuple(uframe[4])
                    vmasks = ufm.ufv_variation_masks_over_stack(uframe[5], uframe[0].shape[1:-2],
                                                                tuple(K) + uframe[0].shape[1:-2])
                    v_sc = _random_variation_sc(uframe, K, rng)                   # garbage padding
                    v_r = [tuple(_ragged_slices(v_sc[j], K, n, lambda c, j=j: _var_shapes(rframes[c])[j])[c]
                                 for j in range(2)) for c in range(n)]
                    out = utvo.utv_corewise_lift(uframe, v_sc + (shape, vmasks), ulf)
                    ref = [tvo.tv_corewise_lift(rframes[c], v_r[c], rlf[c]) for c in range(n)]
                    for j in range(2):
                        _compare(self, out[j], lambda c, j=j: ref[c][j], K, n, 'lift %d' % j)
                    for group in sharing.nontrivial_groups(groups):               # ONE factor direction per group
                        for ii in group[1:]:
                            self.assertTrue(np.array_equal(out[0][ii], out[0][group[0]]))
                    again = utvo.utv_corewise_lift(_dirty_frame(uframe), v_sc + (shape, vmasks), ulf)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(out[:2], again[:2])))
                    y_sc = _random_theta_sc(uframe, K, rng)                       # raw, garbage padding
                    y_r = [tuple(_ragged_slices(y_sc[j], K, n, lambda c, j=j: _theta_shapes(rframes[c])[j])[c]
                                 for j in range(2)) for c in range(n)]
                    tout = utvo.utv_corewise_lift_transpose(uframe, y_sc + _theta_data(uframe)[2:], ulf)
                    tref = [tvo.tv_corewise_lift_transpose(rframes[c], y_r[c], rlf[c]) for c in range(n)]
                    for j in range(2):
                        _compare(self, tout[j], lambda c, j=j: tref[c][j], K, n, 'transpose %d' % j)

    def test_adjoint_identity_on_K_and_C_stacks(self):
        rng = np.random.default_rng(65)
        for name in _SH_DISCRIMINATING:
            for K in ((), (3,), (2, 2)):
                for cd in (0.0, 1e-2, 10.0):
                    with self.subTest(case=name, K=K, curvature_damping=cd):
                        _pts, _rf, uframe, groups, _n = _shared_case(name, 66)
                        ulf = ufo.ufv_tied_lift_factors(uframe, cd, sharing.ufv_shared_frame_data(uframe, groups))
                        shape = tuple(uframe[4])
                        KC = tuple(K) + uframe[0].shape[1:-2]
                        vmasks = ufm.ufv_variation_masks_over_stack(uframe[5], uframe[0].shape[1:-2], KC)
                        v, y = _random_variation_sc(uframe, K, rng), _random_theta_sc(uframe, K, rng)
                        Lv = utvo.utv_corewise_lift(uframe, v + (shape, vmasks), ulf)
                        LTy = utvo.utv_corewise_lift_transpose(uframe, y + _theta_data(uframe)[2:], ulf)
                        mv = ufm.ufv_apply_variations_masks(v + (shape, vmasks))
                        my = ut3mask.ut3_apply_masks(y + _theta_data(uframe)[2:])
                        axes = lambda a: (0,) + tuple(range(1 + len(KC), a.ndim))
                        lhs = sum(np.sum(a * b, axis=axes(a)) for a, b in zip(Lv[:2], my))
                        rhs = sum(np.sum(a * b, axis=axes(a)) for a, b in zip(mv, LTy[:2]))
                        self.assertEqual(np.shape(lhs), KC)                       # one identity per K + C element
                        scale = np.sqrt(sum(np.sum(a * a, axis=axes(a)) for a in Lv[:2])
                                        * sum(np.sum(b * b, axis=axes(b)) for b in my))
                        self.assertTrue(np.all(np.abs(lhs - rhs) <= 1e-12 * scale))

    def test_the_tied_lift_represents_a_tied_tangent(self):
        """``DPhi[L v] == E v`` at eps = 0 per element, for a tied gauged tangent (the uniform shared geometry's own
        projection): the lift, read back on each element's real blocks, represents the same dense tangent through the
        core parametrization at the base point."""
        rng = np.random.default_rng(67)
        for name in _SH_CASES:
            for K in _KS:
                with self.subTest(case=name, K=K):
                    _pts, rframes, uframe, groups, n = _shared_case(name, 68)
                    geom = bgeo.UniformManifoldGeometryOps.from_frame(uframe, (uframe[5][0], uframe[5][2]),
                                                                      _SH_CASES[name][1])
                    aux = geom.precompute(uframe)
                    v = geom.project(uframe, _random_variation_sc(uframe, K, rng), aux)
                    shape = tuple(uframe[4])
                    vmasks = ufm.ufv_variation_masks_over_stack(uframe[5], uframe[0].shape[1:-2],
                                                                tuple(K) + uframe[0].shape[1:-2])
                    th = utvo.utv_corewise_lift(uframe, v + (shape, vmasks),
                                                ufo.ufv_tied_lift_factors(uframe, 0.0, aux))
                    for c in range(n):
                        v_r = tuple(_ragged_slices(v[j], K, n, lambda cc, j=j: _var_shapes(rframes[cc])[j])[c]
                                    for j in range(2))
                        th_r = tuple(_ragged_slices(th[j], K, n, lambda cc, j=j: _theta_shapes(rframes[cc])[j])[c]
                                     for j in range(2))
                        Dth = tvo.tv_to_dense(fvc.t3_corewise_frame((rframes[c][0], rframes[c][2])), th_r)
                        Dv = tvo.tv_to_dense(rframes[c], v_r)
                        self.assertLess(np.linalg.norm(Dth - Dv) / np.linalg.norm(Dv), 1e-12)

    def test_garbage_in_the_companion_padding(self):
        """The padded ``s_g`` levels are zeroed by the masks: finite garbage in every padded slot of the uniform
        companion (``_dirty_companion``) leaves the tied filters, ``sigma_min``, the lift and its transpose bitwise
        unchanged. Negative control: the clip-only filter of the dirty companion is nonzero on the garbage levels,
        and the lift built on it moves."""
        rng = np.random.default_rng(69)
        for name in _SH_DISCRIMINATING:
            with self.subTest(case=name):
                _pts, rframes, uframe, groups, n = _shared_case(name, 70)
                usfd = sharing.ufv_shared_frame_data(uframe, groups)
                dirty = _dirty_companion(usfd, rframes, uframe[1].shape[-2], seed=71)
                self.assertGreater(max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
                                       for a, b in zip(dirty.svd_s, usfd.svd_s)), 1e3)
                clean_lf = ufo.ufv_tied_lift_factors(uframe, 1e-3, usfd)
                dirty_lf = ufo.ufv_tied_lift_factors(uframe, 1e-3, dirty)
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(clean_lf.tied_filters, dirty_lf.tied_filters)))
                self.assertTrue(np.array_equal(bgeo._uniform_lift_sigma_min(clean_lf, uframe[5]),
                                               bgeo._uniform_lift_sigma_min(dirty_lf, uframe[5])))
                K = (2,)
                shape = tuple(uframe[4])
                vmasks = ufm.ufv_variation_masks_over_stack(uframe[5], uframe[0].shape[1:-2],
                                                            K + uframe[0].shape[1:-2])
                v, y = _random_variation_sc(uframe, K, rng), _random_theta_sc(uframe, K, rng)
                for lift in (lambda lf: utvo.utv_corewise_lift(uframe, v + (shape, vmasks), lf),
                             lambda lf: utvo.utv_corewise_lift_transpose(uframe, y + _theta_data(uframe)[2:], lf)):
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(lift(clean_lf)[:2], lift(dirty_lf)[:2])))
                bare = tuple(sharing._tied_filter(dirty, gi, dirty_lf.eps, None) for gi in range(len(dirty.svd_s)))
                self.assertGreater(max(float(np.abs(np.asarray(b) - np.asarray(f)).max())
                                       for b, f in zip(bare, clean_lf.tied_filters)), 1e-6)
                moved = utvo.utv_corewise_lift(uframe, v + (shape, vmasks), dc.replace(dirty_lf, tied_filters=bare))
                self.assertGreater(_pair_relerr(moved[:2], utvo.utv_corewise_lift(uframe, v + (shape, vmasks),
                                                                                  clean_lf)[:2]), 1e-3)

    def test_the_group_clip_is_at_the_real_dimensions(self):
        """The clip decision on the group spectrum: a tied point whose group has a second level at ~1.8e-15 of the
        first -- above the ragged clip ``eps * max(sum_i nD_i, n_g)`` (4 eps, real dims) and below a clip at the
        padded dimensions (16 eps, padding forced to 8) -- keeps that level on both layers: the undamped filter is
        ~4e14 there, ragged and uniform alike (a clip at the padded dimensions would drop it to 0)."""
        np.random.seed(0)
        shape, labels = (5, 5, 7), (0, 0, 1)
        x1, x2 = (t3.TuckerTensorTrain(*sharing.t3_tie_tucker_factors(
            t3.TuckerTensorTrain.randn(shape, (1, 1, 1), (1, 1, 1, 1)).data, labels)) for _ in range(2))
        x = x1 * (1.0 / float(np.linalg.norm(x1.to_dense()))) + x2 * (2.5e-15 / float(np.linalg.norm(x2.to_dense())))
        frame = fvc.t3_orthogonal_representations(sharing.t3_tie_tucker_factors(x.data, labels))[0]
        groups = sharing.validate_sharing(labels, shape)
        sfd = sharing.fv_shared_frame_data(frame, groups)
        eps = np.finfo(float).eps
        ratio = float(sfd.svd_s[0][1] / sfd.svd_s[0][0])
        self.assertTrue(4 * eps < ratio < 16 * eps, ratio)                      # between the two clips
        f = sharing.fv_tied_lift_factors(frame, 0.0, sfd).tied_filters[0]
        uframe = ufc.t3frame_to_ut3frame(frame, N=8, nU=8, nD=8, rL=8, rR=8)
        uf_ = np.asarray(ufo.ufv_tied_lift_factors(uframe, 0.0, sharing.ufv_shared_frame_data(uframe, groups))
                         .tied_filters[0])
        self.assertGreater(float(f[1]), 1e14)                                   # kept by the ragged clip
        self.assertLess(abs(float(uf_[1]) / float(f[1]) - 1.0), 1e-2)          # ... and by the uniform one
        self.assertTrue(np.all(uf_[2:] == 0.0))

    def test_a_companion_of_another_layout_is_a_structural_error(self):
        _pts, rframes, uframe, groups, _n = _shared_case('one group', 72)
        rng = np.random.default_rng(72)
        ragged = sharing.fv_tied_lift_factors(rframes[0], 0.0, sharing.fv_shared_frame_data(rframes[0], groups))
        tied = ufo.ufv_tied_lift_factors(uframe, 0.0, sharing.ufv_shared_frame_data(uframe, groups))
        wrong = dc.replace(tied, shared_data=ragged.shared_data, tied_filters=ragged.tied_filters)
        v = _random_variation_sc(uframe, (), rng) + (tuple(uframe[4]), ufm.ufv_variation_masks(uframe[5]))
        y = _random_theta_sc(uframe, (), rng) + _theta_data(uframe)[2:]
        with self.assertRaisesRegex(ValueError, "companion is not this uniform frame's"):
            utvo.utv_corewise_lift(uframe, v, wrong)
        with self.assertRaisesRegex(ValueError, "companion is not this uniform frame's"):
            utvo.utv_corewise_lift_transpose(uframe, y, wrong)
        with self.assertRaisesRegex(ValueError, 'tied filters'):
            utvo.utv_corewise_lift(uframe, v, dc.replace(tied, tied_filters=()))


def _shared_exact_models(name, seed, rng, geometry='manifold', order=1, W=7, weight=(1.0, 0.5), cd=1e-4,
                         kind='probe_derivatives'):
    """The shared twin of :py:func:`_exact_models`: the uniform shared exact model at the frame packed from the
    ragged ones (its companion built once), the ragged shared exact models per element, the uniform frame, the
    per-element variation shapes, n_elements, and the damping; residual and sample garbage-padded."""
    _pts, rframes, uframe, groups, n = _shared_case(name, seed)
    labels = _SH_CASES[name][1]
    shape = tuple(uframe[4])
    N = uframe[0].shape[-1]
    masks = (uframe[5][0], uframe[5][2])
    ukind, rkind, rsample, sample, rp, r_of = _kind_setup(kind, shape, W, n, rng, masks, N, order, weight)
    if geometry == 'manifold':
        geom = bgeo.UniformManifoldGeometryOps.from_frame(uframe, masks, labels)
        lm = _uniform_exact_model(uframe, masks, ukind, sample, rp, cd, geom=geom)
        rgeom = _MANIFOLD.with_sharing(labels, shape)
        rlm = [_ragged_exact_model(rframes[c], rkind, rsample, r_of(c), cd, geom=rgeom) for c in range(n)]
        vshapes = lambda c: _var_shapes(rframes[c])
    else:
        cframe = ufc.ut3_corewise_frame(_theta_data(uframe))
        geom = bgeo.UniformCorewiseGeometryOps(shape, masks, bgeo.canonical_groups(labels, shape))
        lm = _uniform_exact_model(cframe, masks, ukind, sample, rp, None, geom=geom)
        rcf = [fvc.t3_corewise_frame((f[0], f[2])) for f in rframes]
        rgeom = _COREWISE.with_sharing(labels, shape)
        rlm = [_ragged_exact_model(rcf[c], rkind, rsample, r_of(c), None, geom=rgeom) for c in range(n)]
        vshapes = lambda c: _var_shapes(rcf[c])
        uframe = cframe
    return lm, rlm, uframe, vshapes, n


def _uniform_per_mode_twin(lm, cd):
    """The same uniform exact model with the UNTIED (per-mode) lift factors -- the mutation the tied lift exists to
    prevent (the uniform twin of test_curvature's ``_per_mode_twin``)."""
    data = lm.curvature.geometry_data
    untied = dc.replace(data, lift_factors=ufo.ufv_lift_factors(lm.frame, cd))
    return dc.replace(lm, curvature=bopt.CurvatureState(lm.curvature.misfit_gradient, untied))


def _shared_second_derivative(f, h):
    """``f''(0)`` by the central second difference with one Richardson step (test_curvature's helper)."""
    f0 = f(0.0)
    d2 = lambda s: (f(s) - 2.0 * f0 + f(-s)) / (s * s)
    return (4.0 * d2(0.5 * h) - d2(h)) / 3.0


class TestUniformSharedExactModel(unittest.TestCase):
    """T12's uniform twin, the exact model on the SHARED uniform geometries (T17 rules): every exact action per
    element equal to the ragged shared model's (<= 1e-13; varying-rank stacks, forced padding, garbage-padded
    residual / sample / trial tangents, ``K`` stacks), the outputs tied, exact output masks; garbage in the frame's
    padding (and at TT slack, between the frame's and the point's extents: m29) leaves everything bitwise
    unchanged; the per-mode lift would be caught. Oracles on the uniform Problem path: the second derivative along
    the SHARED uniform retraction (O(h^2), measured as in F14) and the dense Lagrangian Hessian of ``Phi o T``; the
    zero-padded shared restart (the zero ``s_g`` level filtered to exactly 0 on both layers, ``sigma_min`` the real
    zero level); the shared corewise exact Hessian is the per-group mean of the plain one (``project`` around it)."""

    def test_every_exact_action_per_element(self):
        rng = np.random.default_rng(75)
        for geometry in ('manifold', 'corewise'):
            for name in _SH_CASES:
                for K in _KS:
                    with self.subTest(geometry=geometry, case=name, K=K):
                        lm, rlm, uframe, vshapes, n = _shared_exact_models(name, 76, rng, geometry)
                        p = _random_variation_sc(uframe, K, rng)
                        p_r = [tuple(_ragged_slices(p[j], K, n, lambda cc, j=j: vshapes(cc)[j])[c] for j in range(2))
                               for c in range(n)]
                        for label, act in (('hvp', lambda m, q: m.hvp(q)),
                                           ('curvature_hvp', lambda m, q: m.curvature_hvp(q))):
                            out = act(lm, p)
                            refs = [act(rlm[c], p_r[c]) for c in range(n)]
                            for j in range(2):
                                _compare(self, out[j], lambda c, j=j: refs[c][j], K, n, '%s %d' % (label, j))
                            if geometry == 'manifold':                          # the output is TIED
                                vm = ufm.ufv_variation_masks_over_stack(uframe[5], uframe[0].shape[1:-2],
                                                                        tuple(K) + uframe[0].shape[1:-2])
                                res = sharing.ufv_tied_variations_residual(
                                    (out[0], out[1], tuple(uframe[4]), vm), lm.geom_aux)
                                self.assertLess(float(np.max(res)), 1e-12)
                            else:                                               # the per-group mean: one slice
                                for group in lm.geom.groups:
                                    for ii in group[1:]:
                                        self.assertTrue(np.array_equal(out[0][ii], out[0][group[0]]))
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
                        if geometry == 'manifold' and name in _SH_DISCRIMINATING:   # the tied lift is load-bearing
                            pm = _uniform_per_mode_twin(lm, 1e-4).curvature_hvp(p)
                            self.assertGreater(_pair_relerr(pm, lm.curvature_hvp(p)), 1e-3)

    def test_garbage_in_the_frame_padding(self):
        """Garbage in EVERY padded slot of the frame's four supercores (the companion is rebuilt from the dirty frame
        -- masked first) leaves every exact action bit-identical, on both shared uniform geometries."""
        rng = np.random.default_rng(77)
        for geometry in ('manifold', 'corewise'):
            with self.subTest(geometry=geometry):
                lm, _rlm, uframe, _vs, n = _shared_exact_models('ceiling varying', 78, rng, geometry)
                dirty = _dirty_frame(uframe, seed=79)
                if geometry == 'corewise':                     # (U, G, G, G): ONE garbage-padded G, three times
                    dirty = (dirty[0], dirty[2], dirty[2], dirty[2]) + tuple(uframe[4:])
                lmd = _uniform_exact_model(dirty, (uframe[5][0], uframe[5][2]), lm.kind, lm.sample, lm.residual,
                                           1e-4 if geometry == 'manifold' else None, geom=lm.geom)
                p = _random_variation_sc(uframe, (2,), rng)
                for act in (lambda m: m.hvp(p), lambda m: m.curvature_hvp(p), lambda m: m.gradient,
                            lambda m: (m.quadratic(p), m.curvature_quadratic(p))):
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(act(lm), act(lmd))))

    def test_the_frame_masks_at_tt_slack_through_the_model(self):
        """m29 on the shared geometry: at a TIED TT-slack point the kind and the geometry hold the POINT's masks (a
        strict superset of the frame's) while the frame carries garbage in every slot outside its own extent; hvp
        and curvature_hvp are bit-identical to the clean frame's and, per element, the ragged shared model's."""
        rng = np.random.default_rng(81)
        np.random.seed(82)
        labels = (0, 0, 1)
        x = t3.TuckerTensorTrain(*sharing.t3_tie_tucker_factors(
            t3.TuckerTensorTrain.randn((6, 6, 6), (4, 4, 4), (1, 9, 9, 1)).data, labels))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        rframe = fvc.t3_orthogonal_representations(x.data)[0]
        nn, rr = ux.data[0].shape[-2], ux.data[1].shape[-1]
        frame = ufc.t3frame_to_ut3frame(rframe, N=ux.N, nU=nn, nD=nn, rL=rr, rR=rr)
        shape, order, W = x.shape, 1, 6
        ww = [rng.standard_normal((W, N)) for N in shape]
        pp = [rng.standard_normal((W, N)) for N in shape]
        r = [rng.standard_normal((order + 1, W, N)) for N in shape]
        kind = uf.UniformProbeDerivativesKind.from_point(ux.data, order=order, weight=[1.0, 0.5], chunk_size=None)
        sample = uf.pack_sample('probe_derivatives', (ww, pp), ux.N)
        rp = uf.pack_data('probe_derivatives', r, ux.N)
        geom = bgeo.UniformManifoldGeometryOps.from_frame(frame, ux.data[3], labels)    # held masks: the POINT's
        self.assertFalse(np.array_equal(geom.masks[1], frame[5][2]))
        clean = _uniform_exact_model(frame, ux.data[3], kind, sample, rp, 1e-5, geom=geom)
        dirty = _uniform_exact_model(_dirty_frame(frame, seed=83), ux.data[3], kind, sample, rp, 1e-5, geom=geom)
        rkind = bfit.ProbeDerivativesKind(order=order, weight=[1.0, 0.5], chunk_size=None)
        ragged = _ragged_exact_model(rframe, rkind, (ww, pp), r, 1e-5, geom=_MANIFOLD.with_sharing(labels, shape))
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

    def test_padded_shared_restart(self):
        """A zero-padded shared restart that grows ONLY the group's rank (``(2,2,2) -> (3,3,2)``, TT ranks kept, so
        the new ``s_g`` level is the point's only zero singular value): on a frame packed from the ragged one the
        tied filter is the ragged one -- the new real level exactly 0 at every damping, the padded levels exactly 0
        -- and ``sigma_min`` the ragged ~0; on a frame built ON the uniform layer the new level sits at roundoff
        while the padded levels are exactly 0, and ``sigma_min`` reads the REAL level (not a padded 0), whose filter
        is 0; the exact model there is finite, symmetric and, per element, the ragged one."""
        rng = np.random.default_rng(85)
        np.random.seed(85)
        shape, labels = (6, 6, 6), (0, 0, 1)
        groups = sharing.validate_sharing(labels, shape)
        level1 = t3.TuckerTensorTrain(*_shared_unit_point((shape, (2, 2, 2), (1, 2, 2, 1)), labels))
        restart = level1.t3svd(sharing=labels)[0].resize(shape, (3, 3, 2), (1, 2, 2, 1), sharing=labels)
        self.assertTrue(bool(np.all(restart.has_shared_tucker_factors(labels))))
        rframe = fvc.t3_orthogonal_representations(restart.data)[0]
        dims = _frame_dims([rframe])
        uframe = _pack_frames([rframe], dims)
        rsfd, usfd = sharing.fv_shared_frame_data(rframe, groups), sharing.ufv_shared_frame_data(uframe, groups)
        q = rsfd.svd_s[0].shape[-1]                                              # the real level count (ragged)
        self.assertLess(float(rsfd.svd_s[0][-1]), 1e-12 * float(rsfd.svd_s[0][0]))   # the new level: zero
        for cd in (0.0, 1e-6):
            with self.subTest(curvature_damping=cd):
                ulf = ufo.ufv_tied_lift_factors(uframe, cd, usfd)
                rlf = sharing.fv_tied_lift_factors(rframe, cd, rsfd)
                _compare_levels(self, ulf.tied_filters[0], [rlf.tied_filters[0]], 1, 'restart filter', tol=1e-12)
                f = np.asarray(ulf.tied_filters[0])
                self.assertEqual(int(np.sum(f[:q] == 0.0)), 1)                   # the new real level: exactly 0
                self.assertTrue(np.all(f[:q - 1] > 0.0))
                sig_u = float(bgeo._uniform_lift_sigma_min(ulf, uframe[5]))
                self.assertLess(abs(sig_u - float(bgeo._lift_sigma_min(rlf))), 1e-14)
                self.assertLess(sig_u, 1e-12)
        ux = ut3.UniformTuckerTensorTrain.from_t3(restart, N=dims[0], n=dims[1], r=dims[3])
        uframe2 = ufc.ut3_orthogonal_representations(ux.data)[0]                 # built on the uniform layer
        usfd2 = sharing.ufv_shared_frame_data(uframe2, groups)
        lf2 = ufo.ufv_tied_lift_factors(uframe2, 0.0, usfd2)
        s2 = np.asarray(usfd2.svd_s[0])
        self.assertGreater(s2.shape[-1], q)                                      # padded levels exist
        # the smallest lift singular value is the new REAL level (observed 2.4e-31 here, the padded ones 0.0), never
        # a padded one; its filter is exactly 0
        self.assertEqual(float(bgeo._uniform_lift_sigma_min(lf2, uframe2[5])), float(np.min(s2[:q])))
        self.assertEqual(float(np.asarray(lf2.tied_filters[0])[q - 1]), 0.0)
        self.assertTrue(np.all(np.asarray(lf2.tied_filters[0])[q:] == 0.0))
        ww = [rng.standard_normal((8, N)) for N in shape]
        pp = [rng.standard_normal((8, N)) for N in shape]
        r = [rng.standard_normal((2, 8, N)) for N in shape]
        N = uframe[0].shape[-1]
        masks = (uframe[5][0], uframe[5][2])
        kind = uf.UniformProbeDerivativesKind.from_point((None, None, shape, masks), order=1, chunk_size=None)
        geom = bgeo.UniformManifoldGeometryOps.from_frame(uframe, masks, labels)
        lm = _uniform_exact_model(uframe, masks, kind, (ut3o.pack_if_ragged(ww, N), ut3o.pack_if_ragged(pp, N)),
                                  ut3o.pack_if_ragged(r, N), None, geom=geom)
        rlm = _ragged_exact_model(rframe, bfit.ProbeDerivativesKind(order=1, chunk_size=None), (ww, pp), r, None,
                                  geom=_MANIFOLD.with_sharing(labels, shape))
        P = _random_variation_sc(uframe, (3,), rng)
        HP = lm.hvp(P)
        self.assertTrue(all(np.all(np.isfinite(a)) for a in HP))
        P_r = tuple(_ragged_slices(P[j], (3,), 1, lambda c, j=j: _var_shapes(rframe)[j])[0] for j in range(2))
        ref = rlm.hvp(P_r)
        for j in range(2):
            _compare(self, HP[j], lambda c, j=j: ref[j], (3,), 1, 'restart hvp %d' % j, tol=1e-12)
        PP = geom.project(uframe, P, lm.geom_aux)
        vm = ufm.ufv_variation_masks_over_stack(uframe[5], (), (3,))
        a = ufm.ufv_apply_variations_masks((PP[0], PP[1], shape, vm))
        b = ufm.ufv_apply_variations_masks(tuple(lm.hvp(PP)) + (shape, vm))
        gram = sum(np.einsum('dkx,dlx->kl', s.reshape(s.shape[:2] + (-1,)), t.reshape(t.shape[:2] + (-1,)))
                   for s, t in zip(a, b))
        self.assertLess(np.abs(gram - gram.T).max(), 1e-12 * np.abs(gram).max())

    def test_quadratic_form_is_the_second_derivative_along_the_shared_retraction(self):
        """On the uniform Problem path (``uniform_least_squares_problem(..., sharing=)``, its own frame):
        ``<p, H p> == d^2/dt^2 f(R_sh(t p))`` along the SHARED uniform retraction to the finite difference's
        resolution, which also separates it from Gauss-Newton and from the per-mode-lift twin; orders 0-2."""
        rng = np.random.default_rng(87)
        cases = {'ceiling': ((((6, 6, 4), (4, 4, 2), (1, 2, 2, 1)), (0, 0, 1)), 1, [1.0, 0.5]),
                 'two_groups': ((((5, 6, 5, 6), (3, 2, 3, 2), (1, 2, 3, 2, 1)), (0, 1, 0, 1)), 0, None),
                 'one_group': ((((5, 5, 5), (3, 3, 3), (1, 2, 2, 1)), (0, 0, 0)), 2, [1.0, 0.5, 0.25])}
        for name, ((struct, labels), order, weight) in cases.items():
            for cd in (0.0, None):
                with self.subTest(case=name, curvature_damping=cd):
                    np.random.seed(88)
                    x, other = _shared_unit_point(struct, labels), _shared_unit_point(struct, labels)
                    ww = [rng.standard_normal((12, N)) for N in struct[0]]
                    pp = [rng.standard_normal((12, N)) for N in struct[0]]
                    data = [z + 0.1 * rng.standard_normal(z.shape) for z in sd.t3_probe_derivatives(ww, pp, other,
                                                                                                    order)]
                    ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*x))
                    prob = uf.uniform_least_squares_problem('manifold', 'probe_derivatives', ux, (ww, pp), data,
                                                            order=order, weight=weight, chunk_size=None,
                                                            sharing=labels)
                    lm = prob.local_model((ux.data[0], ux.data[1]), hessian='exact', curvature_damping=cd)
                    v = prob.geom.project(lm.frame, _random_variation_sc(lm.frame, (), rng), lm.geom_aux)
                    v = cw.corewise_scale(v, 1.0 / float(prob.geom.inner(v, v)) ** 0.5)
                    q, q_gn = float(lm.quadratic(v)), float(lm.gn_quadratic(v))
                    self.assertLess(abs(float(prob.geom.inner(v, lm.hvp(v))) - q), 1e-12 * abs(q))
                    h = 0.05 * min(float(lm.curvature.geometry_data.sigma_min), 1.0)
                    fd = _shared_second_derivative(
                        lambda t: float(prob.objective(lm.retract(cw.corewise_scale(v, t)))), h)
                    self.assertLess(abs(fd - q), 1e-5 * max(abs(q), abs(q_gn)))
                    self.assertLess(abs(fd - q), 1e-3 * abs(q - q_gn))
                    if cd == 0.0:
                        q_pm = float(_uniform_per_mode_twin(lm, 0.0).quadratic(v))
                        self.assertGreater(abs(fd - q_pm), 1e3 * abs(fd - q))

    def test_the_shared_retraction_is_second_order_here(self):
        """F14 (``c15_shared_exact``'s set-up, seed 5) on the uniform layer: the plain central second difference along
        the SHARED uniform retraction converges to the exact model's ``<v, H v>`` at O(h^2) -- the error falls ~4x
        per halving -- while Gauss-Newton is off by far more (the values depend on the frame's SVD signs, so only the
        convergence is asserted, as on the ragged layer)."""
        np.random.seed(5)
        rng = np.random.default_rng(5)
        shape, labels = (5, 5, 6), (0, 0, 1)
        x = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
        x = x.share(labels, max_tucker_ranks=(2, 2, 2), max_tt_ranks=(1, 2, 2, 1))
        x = x.rank_adjustment_sweep('right_to_left', sharing=labels)
        ww = [rng.standard_normal((60, N)) for N in shape]
        pp = [rng.standard_normal((60, N)) for N in shape]
        data = [z + 0.3 * rng.standard_normal(z.shape) for z in sd.t3_probe_derivatives(ww, pp, x.data, 2)]
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        prob = uf.uniform_least_squares_problem('manifold', 'probe_derivatives', ux, (ww, pp), data, order=2,
                                                chunk_size=None, sharing=labels)
        sc = (ux.data[0], ux.data[1])
        lm = prob.local_model(sc, hessian='exact', curvature_damping=0.0)
        v = prob.geom.project(lm.frame, _random_variation_sc(lm.frame, (), rng), lm.geom_aux)
        v = cw.corewise_scale(v, 1.0 / float(prob.geom.inner(v, v)) ** 0.5)
        q, q_gn = float(lm.quadratic(v)), float(lm.gn_quadratic(v))
        f0 = float(prob.objective(sc))
        errors = []
        for h in (2e-2, 1e-2, 5e-3):
            fp = float(prob.objective(lm.retract(cw.corewise_scale(v, h))))
            fm = float(prob.objective(lm.retract(cw.corewise_scale(v, -h))))
            errors.append(abs((fp - 2.0 * f0 + fm) / h ** 2 - q))
        for e_h, e_half in zip(errors, errors[1:]):
            self.assertTrue(3.0 < e_h / e_half < 5.0, errors)
        self.assertLess(errors[-1], 1e-3 * abs(q - q_gn))

    def test_against_the_dense_lagrangian_hessian(self):
        """``Pi_sh L^T H_lag L Pi_sh`` with ``H_lag`` the exact dense core-coordinate Hessian of the Lagrangian
        ``f(Phi(theta)) - <grad_sh f, Phi(theta)>`` and ``L`` the INDEPENDENT dense tied lift (pinv of the stacked
        ``S_i^T``; tests/backend/_curvature_oracles.py) at the ragged frame -- against the uniform shared model at
        that frame packed with padding (garbage-padded tangent, residual and sample), read back on the real blocks;
        a group-ceiling structure, where the per-mode lift is off by orders of magnitude."""
        rng = np.random.default_rng(89)
        np.random.seed(89)
        struct, labels, order, weight, W = ((4, 4, 3), (3, 3, 2), (1, 2, 2, 1)), (0, 0, 1), 2, [1.0, 0.5, 0.25], 4
        shape = struct[0]
        x, other = _shared_unit_point(struct, labels), _shared_unit_point(struct, labels)
        ww = [rng.standard_normal((W, N)) for N in shape]
        pp = [rng.standard_normal((W, N)) for N in shape]
        data = [z + 0.1 * rng.standard_normal(z.shape) for z in sd.t3_probe_derivatives(ww, pp, other, order)]
        rgeom = _MANIFOLD.with_sharing(labels, shape)
        rprob = bopt.least_squares_problem(rgeom, bfit.ProbeDerivativesKind(order=order, weight=weight,
                                                                            chunk_size=None), (ww, pp), data)
        rlm = rprob.local_model(x, hessian='exact', curvature_damping=0.0)
        rframe, aux = rlm.frame, rlm.geom_aux
        uframe = _pack_frames([rframe], _frame_dims([rframe]))
        N = uframe[0].shape[-1]
        masks = (uframe[5][0], uframe[5][2])
        ukind = uf.UniformProbeDerivativesKind.from_point((None, None, shape, masks), order=order, weight=weight,
                                                          chunk_size=None)
        usample = (_dirty_packed_vectors(ut3o.pack_if_ragged(ww, N), shape, 90),
                   _dirty_packed_vectors(ut3o.pack_if_ragged(pp, N), shape, 91))
        ur = _dirty_packed_vectors(ut3o.pack_if_ragged(rlm.residual, N), shape, 92)
        ulm = _uniform_exact_model(uframe, masks, ukind, usample, ur, 0.0,
                                   geom=bgeo.UniformManifoldGeometryOps.from_frame(uframe, masks, labels))
        omega = rprob.kind.weight

        def measure(cores):                                               # omega (.) S(X), flattened
            out = sd.t3_probe_derivatives(ww, pp, cores, order)
            return np.concatenate([(z * omega[i if omega.shape[0] > 1 else 0].reshape(-1, 1, 1)).ravel()
                                   for i, z in enumerate(out)])
        cols = []
        for idx in np.ndindex(*shape):
            unit = (tuple(np.eye(Ni)[j][None, :] for Ni, j in zip(shape, idx)),
                    tuple(np.ones((1, 1, 1)) for _ in shape))
            cols.append(measure(unit))
        A = np.stack(cols, axis=1)
        y = A @ np.asarray(t3c.t3_to_dense(x)).ravel() - np.concatenate(
            [(r * omega[i if omega.shape[0] > 1 else 0].reshape(-1, 1, 1)).ravel() for i, r in enumerate(rlm.residual)])
        theta = (tuple(rframe[0]), tuple(rframe[2]))
        G = tvo.tv_to_dense(rframe, rlm.curvature.misfit_gradient)              # grad_sh f, dense
        groups = sharing.validate_sharing(labels, shape)
        for _ in range(2):
            p = _random_variation_sc(uframe, (), rng)                            # garbage padding
            p_r = tuple(_ragged_slices(p[j], (), 1, lambda c, j=j: _var_shapes(rframe)[j])[0] for j in range(2))
            pg = rgeom.project(rframe, p_r, aux)
            H_theta = oracles.dense_lagrangian_hessian(A, y, theta, oracles.tied_lift(rframe, groups, pg), G)
            ref = rgeom.project(rframe, oracles.tied_lift_transpose(rframe, groups, H_theta), aux)
            got = ulm.hvp(p)
            err = max(_compare(self, got[j], lambda c, j=j: ref[j], (), 1, 'hvp %d' % j, tol=1e-10) for j in range(2))
            pm = _uniform_per_mode_twin(ulm, 0.0).hvp(p)
            pm_err = max(np.linalg.norm(np.asarray(pm[j])[(i,) + tuple(slice(0, s) for s in R.shape)] - R)
                         / np.linalg.norm(R) for j in range(2) for i, R in enumerate(ref[j]))
            self.assertGreater(pm_err, 1e4 * max(err, 1e-16))

    def test_m27_the_exact_gradient_is_the_gauss_newton_gradient(self):
        """Through ``uniform_least_squares_problem(..., sharing=)``: the shared exact model's gradient is the
        Gauss-Newton model's (the TIED gradient), bit for bit, on both shared uniform geometries."""
        rng = np.random.default_rng(93)
        np.random.seed(94)
        struct, labels = ((6, 6, 4), (4, 4, 2), (1, 2, 2, 1)), (0, 0, 1)
        ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain(*_shared_unit_point(struct, labels)))
        ww, pp, _ = _pd_sample(struct[0], 9, 2, 1, rng)
        data = sd.t3_probe_derivatives(ww, pp, _shared_unit_point(struct, labels), 2)
        for geometry in ('manifold', 'corewise'):
            with self.subTest(geometry=geometry):
                prob = uf.uniform_least_squares_problem(geometry, 'probe_derivatives', ux, (ww, pp), data, order=2,
                                                        weight=[1.0, 0.5, 0.25], sharing=labels)
                sc = (ux.data[0], ux.data[1])
                gn, ex = prob.local_model(sc), prob.local_model(sc, hessian='exact')
                self.assertTrue(ex.is_exact)
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(gn.gradient, ex.gradient)))


def _shared_restart_template(shape, level1, restart, target, W, seed=7):
    """T22's template (tests/backend/test_curvature.py ``_shared_restart_problem``), parametrized by structure:
    probe derivatives of order 2, omega = [1, .5, .25], 5 % noise, a TIED target, a shared fit at the level-1 ranks
    (30 Gauss-Newton steps) zero-padded to the restart ranks with ``resize(..., sharing=)``. Returns the ragged
    problem, the sample, the data, the weight and the tied restart."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    labels = (0, 0, 1)
    tgt = t3.TuckerTensorTrain(*sharing.t3_tie_tucker_factors(t3.TuckerTensorTrain.randn(shape, *target).data, labels))
    ww = [rng.standard_normal((W, N)) for N in shape]
    pp = [rng.standard_normal((W, N)) for N in shape]
    weight = np.array([1.0, 0.5, 0.25])
    kind = bfit.probe_derivatives_kind(2, weight)
    data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(tgt.data, (ww, pp))]
    problem = bopt.least_squares_problem(_MANIFOLD.with_sharing(labels, shape), kind, (ww, pp), data)
    x0 = sharing.t3_tie_tucker_factors(t3.TuckerTensorTrain.randn(shape, *level1).data, labels)
    x1, _ = bopt.newton_cg(problem, x0, max_newton=30, gtol_rel=1e-10)
    return problem, (ww, pp), data, weight, t3.TuckerTensorTrain(*x1).resize(shape, *restart, sharing=labels)


_T22 = ((6, 6, 6), ((2, 2, 2), (1, 2, 2, 1)), ((3, 3, 3), (1, 3, 3, 1)), ((4, 4, 4), (1, 4, 4, 1)), 60)
_CEILING_RESTART = ((7, 7, 4), ((3, 3, 2), (1, 2, 2, 1)), ((4, 4, 2), (1, 2, 2, 1)), ((5, 5, 3), (1, 3, 3, 1)), 60)


def _uniform_shared_problem(problem, sample, data, weight, x):
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    uprob = uf.uniform_least_squares_problem('manifold', 'probe_derivatives', ux, sample, data, order=2, weight=weight,
                                             chunk_size=problem.kind.chunk_size, sharing=(0, 0, 1))
    return uprob, ux


class TestUniformSharedExactNewton(unittest.TestCase):
    """T22 on the SHARED uniform layer, to the equivalence contract (``docs/uniform_equivalence_contract.md``
    §"The full-rank qualifier", S8's T21 ruling). From a zero-padded shared restart the start is rank-DEFICIENT and
    the two layers' frames legitimately pick different completions, so their trajectories differ by design: the
    uniform run must meet T22 itself -- converge on the gradient test with at most two rejections, its first row
    reporting the zero ``s_g`` level -- reach the ragged run's minimizer, and stay tied; the frontend run IS the
    backend run. From a numerically FULL-RANK shared start the uniform run IS the ragged run (the same outer /
    accepted / CG counts, the objective row by row). Observed (S9): T22 ragged 10 / 0 / 41 outer / rejected / CG,
    uniform 13 / 0 / 40, the same final objective to the last bit. The group-ceiling restart (``(7,7,4)``, shared
    rank ``(3,3,2) -> (4,4,2)``, target ``(5,5,3)/(1,3,3,1)``; mode 0's ``S_0^T`` wide) discriminates the lift end to
    end: uniform tied 7 / 0 / 24 (ragged 7 / 0 / 28), the per-mode lift 12 / 0 / 59; from a full-rank start at those
    ranks the uniform run is the ragged run, 9 / 0 / 28 (per-mode 14 / 0 / 63)."""

    def test_zero_padded_shared_restart(self):
        problem, sample, data, weight, restart = _shared_restart_template(*_T22)
        _, ragged = bopt.newton_cg(problem, restart.data, max_newton=40, hessian='exact')
        self.assertEqual(ragged['stop_reason'], 'gradient')
        uprob, ux = _uniform_shared_problem(problem, sample, data, weight, restart)
        xu, ub = bopt.newton_cg(uprob, (ux.data[0], ux.data[1]), max_newton=40, hessian='exact')
        h = ub['history']
        self.assertEqual(ub['stop_reason'], 'gradient')
        self.assertLessEqual(sum(row['tr_accepted'] is False for row in h), 2)
        self.assertLess(h[0]['lift_sigma_min'], 1e-12)                       # the zero s_g level: a singular tied lift
        self.assertTrue(all(np.isfinite(row['pKp']) for row in h if not row['converged']))
        self.assertLessEqual(ub['newton_steps'], 16)
        self.assertLess(abs(ub['losses'][-1] - ragged['losses'][-1]) / ragged['losses'][-1], 1e-9)   # same minimizer
        self.assertTrue(bool(np.all(sharing.ut3_tucker_factors_shared((xu[0], xu[1]) + ux.data[2:], (0, 0, 1)))))

    def test_group_ceiling_restart_backend_and_frontend(self):
        import t3toolbox.optimizers as topt
        import t3toolbox.shared_geometry as sg
        import t3toolbox.uniform_manifold as ut3m
        problem, sample, data, weight, restart = _shared_restart_template(*_CEILING_RESTART)
        _, ragged = bopt.newton_cg(problem, restart.data, max_newton=40, hessian='exact')
        uprob, ux = _uniform_shared_problem(problem, sample, data, weight, restart)
        xb, ub = bopt.newton_cg(uprob, (ux.data[0], ux.data[1]), max_newton=40, hessian='exact')
        self.assertEqual(ub['stop_reason'], 'gradient')
        self.assertLessEqual(sum(row['tr_accepted'] is False for row in ub['history']), 2)
        self.assertLessEqual(ub['newton_steps'], 9)                          # the per-mode lift needs 12 here
        self.assertLess(abs(ub['losses'][-1] - ragged['losses'][-1]) / ragged['losses'][-1], 1e-9)
        xf, uf_ = topt.newton_cg(sg.shared(ut3m.UNIFORM_MANIFOLD, (0, 0, 1)), 'probe_derivatives', sample, data, ux,
                                 order=2, weight=weight, max_newton=40, hessian='exact',
                                 chunk_size=problem.kind.chunk_size)
        self.assertEqual(uf_['losses'], ub['losses'])                       # the frontend IS the backend run
        self.assertEqual([row['cg_iters'] for row in uf_['history']], [row['cg_iters'] for row in ub['history']])
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(xf.data[:2], xb)))
        self.assertTrue(bool(np.all(xf.has_shared_tucker_factors((0, 0, 1)))))

    def test_group_ceiling_full_rank_start_is_the_ragged_run(self):
        shape, _level1, restart_ranks, target, W = _CEILING_RESTART
        rng = np.random.default_rng(7)
        np.random.seed(7)
        labels = (0, 0, 1)
        tgt = t3.TuckerTensorTrain(*sharing.t3_tie_tucker_factors(t3.TuckerTensorTrain.randn(shape, *target).data,
                                                                   labels))
        ww = [rng.standard_normal((W, N)) for N in shape]
        pp = [rng.standard_normal((W, N)) for N in shape]
        weight = np.array([1.0, 0.5, 0.25])
        kind = bfit.probe_derivatives_kind(2, weight)
        data = [z + 0.05 * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
                for z in kind.point_forward(tgt.data, (ww, pp))]
        problem = bopt.least_squares_problem(_MANIFOLD.with_sharing(labels, shape), kind, (ww, pp), data)
        np.random.seed(3)
        x0 = t3.TuckerTensorTrain(*sharing.t3_tie_tucker_factors(t3.TuckerTensorTrain.randn(shape, *restart_ranks).data,
                                                                 labels)).t3svd(sharing=labels)[0]
        _, ragged = bopt.newton_cg(problem, x0.data, max_newton=40, hessian='exact')
        self.assertEqual(ragged['stop_reason'], 'gradient')
        uprob, ux = _uniform_shared_problem(problem, (ww, pp), data, weight, x0)
        _, ub = bopt.newton_cg(uprob, (ux.data[0], ux.data[1]), max_newton=40, hessian='exact')
        _same_run(self, ub, ragged)


if __name__ == '__main__':
    unittest.main()
