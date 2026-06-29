# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
"""Cheap jax-dispatch tests.

The backend is written backend-agnostically (``xnp``, ``'...'`` einsums, inferred dispatch), so the
*numerical* answer is identical in numpy and jax -- it is fair to trust jax's numerics and not
duplicate every numerical sweep. What we DO need to check is that jax is actually invoked: that a jax
input dispatches to a pure-jax computation with no hidden numpy calls.

``jax.jit`` is the strong proof: a stray ``np.*`` call on a tracer raises, so a function that
jit-compiles and returns jax leaves is pure-jax. Dynamic-shape ops (rtol/atol truncation choose ranks
from the data) cannot be jitted, so they get a weaker jax-in -> jax-out output check. A handful of
numerical smoke tests guard against subtle backend divergence on the most complex ops.
"""
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.common as common
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.probe_derivatives as pd
import t3toolbox.backend.tangent_operations as tops
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.orthogonal_representations as orth_reps
import t3toolbox.backend.probing as probing
import t3toolbox.fitting as fitting

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

STRUCT = ((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))  # small -> fast compiles
tol = 1e-9
norm = np.linalg.norm


@unittest.skipUnless(HAS_JAX, "jax not available")
class TestDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.x_np = t3.TuckerTensorTrain.randn(*STRUCT)
        cls.x = cls.x_np.to_jax()
        cls.base, cls.var = bvf.t3_orthogonal_representations(cls.x)
        cls.v = t3m.T3Tangent(cls.base, cls.var)
        cls.w = t3m.COREWISE.randn(cls.base)
        cls.v_vstack = t3m.COREWISE.randn(cls.base, stack_shape=(3,))  # K=(3,)
        cls.ww = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])  # probe stack W=(2,)
        cls.zz = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])  # W + C + (N,), C=()
        cls.zz_vstack = tuple(jnp.array(np.random.randn(2, 3, N)) for N in STRUCT[0])  # W + K + C, K=(3,)
        cls.x_other = t3.TuckerTensorTrain.randn((4, 5, 6), (3, 3, 3), (1, 2, 2, 1)).to_jax()
        # uniform fixtures: a jax UT3 (supercores jax, masks numpy/host -- slice 7) + a second one to add/inner
        cls.ux = ut3.UniformTuckerTensorTrain.from_t3(cls.x_np).to_jax()
        cls.uy = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(*STRUCT)).to_jax()
        cls.uvecs = tuple(jnp.array(np.random.randn(N)) for N in STRUCT[0])
        cls.uww = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])
        cls.uidx = jnp.array([1, 2, 3])

    # ---------------------------------------------------------------- helpers
    def _leaves_all_jax(self, out):
        leaves = jax.tree_util.tree_leaves(out)
        self.assertTrue(leaves, "output has no array leaves to check")
        for leaf in leaves:
            self.assertTrue(common.is_jax_ndarray(leaf),
                            "non-jax leaf %s -- numpy leaked into the computation" % type(leaf))

    def assert_jit_jax(self, fn, *args):
        """jit-compile fn (proves no numpy on tracers) and check every output leaf is a jax array."""
        self._leaves_all_jax(jax.jit(fn)(*args))

    def assert_eager_jax(self, fn, *args):
        """For dynamic-shape ops that can't jit: just check jax in -> jax out (eager)."""
        self._leaves_all_jax(fn(*args))

    def assert_concrete_masks(self, ut):
        """A jitted uniform op that RETURNS a ut3 must keep its masks CONCRETE (host numpy), not tracers.

        ``_leaves_all_jax`` checks only the children (supercores); the masks ride as aux_data and are
        skipped by tree_leaves -- so a mask-recomputing op (orthogonalize/svd/+) could silently leak a
        tracer mask into aux_data (the slice-7 failure mode). This catches that.
        """
        for m in ut.masks.data:
            self.assertFalse(isinstance(m, jax.core.Tracer),
                             "uniform op leaked a TRACER mask into aux_data (must stay host/numpy)")
            self.assertTrue(common.is_numpy_ndarray(m), "uniform mask must be a host numpy array")

    def assert_jit_uniform(self, fn, *args, returns_ut3=False):
        """jit-compile a uniform op; check jax leaves; if it returns a ut3, check the masks stay concrete."""
        out = jax.jit(fn)(*args)
        self._leaves_all_jax(out)
        if returns_ut3:
            self.assert_concrete_masks(out)

    # ---------------------------------------------------- jit bucket: TuckerTensorTrain
    def test_jit_tucker_tensor_train(self):
        vecs = tuple(jnp.ones(N) for N in STRUCT[0])
        idx = jnp.array([1, 2, 3])
        self.assert_jit_jax(lambda a: a.to_dense(), self.x)
        self.assert_jit_jax(lambda a, b: a.inner(b), self.x, self.x_other)  # T3-T3 inner (t3_linalg zipper)
        self.assert_jit_jax(lambda a, v0, v1, v2: a.apply((v0, v1, v2)), self.x, *vecs)
        self.assert_jit_jax(lambda a, i: a.entries(i), self.x, idx)
        self.assert_jit_jax(  # t3svd with FIXED ranks -> static shapes -> jit-able
            lambda a: a.t3svd(max_tucker_ranks=(2, 2, 2), max_tt_ranks=(1, 2, 2, 1)), self.x)
        self.assert_jit_jax(  # rank_adjustment_sweep: directional rank-drop, static shapes -> jit-able
            lambda a: a.t3svd(max_tt_ranks=(1, 2, 2, 1))[0].rank_adjustment_sweep('right_to_left'), self.x)
        # t3m methods with FIXED max-ranks -> static shapes -> jit-able (rtol/atol stay eager)
        for mth in ('form_then_round', 'inplace_fused', 'swap'):
            self.assert_jit_jax(
                lambda a, b, m=mth: a.t3m(b, method=m, max_tucker_ranks=2, max_tt_ranks=2),
                self.x, self.x_other)
        self.assert_jit_jax(  # swap + oversample -> t3svd cleanup at fixed ranks, still static
            lambda a, b: a.t3m(b, method='swap', max_tucker_ranks=2, max_tt_ranks=2, oversample=2),
            self.x, self.x_other)
        self.assert_jit_jax(lambda a, b: a + b, self.x, self.x_other)  # T3+T3 (t3_add)
        self.assert_jit_jax(lambda a, b: a - b, self.x, self.x_other)  # T3-T3
        self.assert_jit_jax(lambda a: a + 2.5, self.x)                 # T3+scalar (t3_plus_scalar)
        self.assert_jit_jax(  # resize (change_tucker/tt_core_shapes); static shapes -> jit-able
            lambda a: a.resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1)), self.x)
        # ambient adjoints: residual c shape W (+ C); both sum modes; returns CP factors (a pytree)
        N = STRUCT[0]
        self.assert_jit_jax(
            lambda cc, *ws: t3.TuckerTensorTrain.apply_ambient_transpose(cc, ws), jnp.ones(2), *self.ww)
        self.assert_jit_jax(
            lambda cc, *ws: t3.TuckerTensorTrain.apply_ambient_transpose(cc, ws, sum_over_probes=True),
            jnp.ones(2), *self.ww)
        self.assert_jit_jax(
            lambda cc, i: t3.TuckerTensorTrain.entries_ambient_transpose(cc, i, N), jnp.ones(()), idx)
        self.assert_jit_jax(
            lambda cc, i: t3.TuckerTensorTrain.entries_ambient_transpose(cc, i, N, sum_over_probes=True),
            jnp.ones(2), jnp.array([[1, 2], [2, 3], [3, 0]]))  # (d,) + W, W=(2,)
        # corewise (non-manifold) adjoints: gradient w.r.t. the cores (base x passed as a traced pytree)
        self.assert_jit_jax(
            lambda xx, cc, *ws: xx.apply_corewise_transpose(cc, ws), self.x, jnp.ones(2), *self.ww)
        self.assert_jit_jax(
            lambda xx, cc, *ws: xx.apply_corewise_transpose(cc, ws, sum_over_probes=True),
            self.x, jnp.ones(2), *self.ww)
        self.assert_jit_jax(
            lambda xx, cc, i: xx.entries_corewise_transpose(cc, i), self.x, jnp.ones(()), idx)
        self.assert_jit_jax(
            lambda xx, cc, i: xx.entries_corewise_transpose(cc, i, sum_over_probes=True),
            self.x, jnp.ones(2), jnp.array([[1, 2], [2, 3], [3, 0]]))  # (d,) + W, W=(2,)
        # probe transposes: residuals zz (d vecs, W+C+(N,)), probe vecs ww; ambient -> CP, corewise -> grads
        self.assert_jit_jax(
            lambda zt, w: t3.TuckerTensorTrain.probe_ambient_transpose(zt, w), list(self.zz), list(self.ww))
        self.assert_jit_jax(
            lambda zt, w: t3.TuckerTensorTrain.probe_ambient_transpose(zt, w, sum_over_probes=True),
            list(self.zz), list(self.ww))
        self.assert_jit_jax(
            lambda xx, zt, w: xx.probe_corewise_transpose(zt, w), self.x, list(self.zz), list(self.ww))
        self.assert_jit_jax(
            lambda xx, zt, w: xx.probe_corewise_transpose(zt, w, sum_over_probes=True),
            self.x, list(self.zz), list(self.ww))

    # ---------------------------------------------------- jit bucket: T3Tangent
    def test_jit_tangent(self):
        base = self.base  # close over the fixed base (aux_data); never a traced arg
        self.assert_jit_jax(lambda a: a.to_dense(), self.v)
        self.assert_jit_jax(lambda a: a.to_t3(), self.v)
        self.assert_jit_jax(lambda a: t3m.MANIFOLD.retract(a), self.v)
        self.assert_jit_jax(lambda a, b: a.corewise_inner(b), self.v, self.w)   # binary op; same-frame guard skips under the trace
        self.assert_jit_jax(lambda a: a.corewise_norm(), self.v)
        self.assert_jit_jax(lambda a, b: a + b, self.v, self.w)
        self.assert_jit_jax(lambda a, b: a - b, self.v, self.w)
        self.assert_jit_jax(lambda a: 2.5 * a, self.v)
        self.assert_jit_jax(lambda a: t3m.MANIFOLD.project(a), self.v)
        self.assert_jit_jax(lambda a: t3m.MANIFOLD.project_oblique(a), self.v)
        self.assert_jit_jax(lambda xx: t3m.MANIFOLD.project_ambient(base, xx), self.x_other)
        self.assert_jit_jax(lambda a, w: a.probe(w), self.v, self.ww)
        self.assert_jit_jax(lambda a, w: a.probe(w), self.v_vstack, self.ww)  # 3-group (W,K,C) probe
        self.assert_jit_jax(lambda z, w: t3m.T3Tangent.probe_transpose(z, w, base), self.zz, self.ww)
        # K-stacked residuals (W+K+C) -> 3-group transpose assemble, both sum modes
        self.assert_jit_jax(
            lambda z, w: t3m.T3Tangent.probe_transpose(z, w, base, sum_over_probes=True), self.zz_vstack, self.ww)
        self.assert_jit_jax(
            lambda z, w: t3m.T3Tangent.probe_transpose(z, w, base), self.zz_vstack, self.ww)
        self.assert_jit_jax(lambda a, w: a.apply(w), self.v_vstack, self.ww)              # tangent apply
        self.assert_jit_jax(lambda a, i: a.entries(i), self.v_vstack, jnp.array([1, 2, 3]))  # tangent entries
        # tangent adjoints: c shape W (+C); both sum modes
        self.assert_jit_jax(
            lambda cc, w: t3m.T3Tangent.apply_transpose(cc, w, base, sum_over_probes=True), jnp.ones(2), self.ww)
        self.assert_jit_jax(
            lambda cc, w: t3m.T3Tangent.apply_transpose(cc, w, base), jnp.ones(2), self.ww)  # keep W
        self.assert_jit_jax(
            lambda cc, i: t3m.T3Tangent.entries_transpose(cc, i, base, sum_over_probes=True),
            jnp.ones(()), jnp.array([1, 2, 3]))

    # ---------------------------------------------------- jit bucket: symmetric probe derivatives
    def test_jit_probe_derivatives(self):
        # paired (X, P) sample stack W=(2,); order static; all-orders jet output must be all-jax.
        self.assert_jit_jax(
            lambda cc, w, p: pd.probe_derivatives_t3(w, p, cc, 3),
            self.x.data, list(self.ww), list(self.zz))
        # with a base/core stack C=(2,) too: output (K+1) + W + C + (N,)
        xc = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=(2,)).to_jax()
        self.assert_jit_jax(
            lambda cc, w, p: pd.probe_derivatives_t3(w, p, cc, 3),
            xc.data, list(self.ww), list(self.zz))
        # the new t-contractions directly (order axis t=3 leading; W=(2,), C=())
        trs = pd.binomial_combine_tensor(3)
        mu = jnp.ones((4, 2, 5)); G = jnp.ones((5, 4, 6)); xij = jnp.ones((2, 2, 4)); nu = jnp.ones((4, 2, 6))
        self.assert_jit_jax(lambda a, b, c, e: contractions.trs_rWCa_Caib_sWCi_to_tWCb(a, b, c, e),
                            trs[:, :, :2], mu, G, xij)
        self.assert_jit_jax(lambda a, b, c, e: contractions.trs_rWCa_Caib_sWCb_to_tWCi(a, b, c, e),
                            trs, mu, G, nu)
        eta = jnp.ones((4, 2, 4)); U = jnp.ones((4, 7))
        self.assert_jit_jax(lambda a, b: contractions.tWCi_Cio_to_tWCo(a, b), eta, U)
        # Riemannian (tangent) forward derivatives: jit, base + variation sweeps, all-orders
        self.assert_jit_jax(
            lambda var, b, w, p: pd.probe_tangent_derivatives(w, p, var, b, 3),
            self.var.data, self.base.data, list(self.ww), list(self.zz))
        # Riemannian (tangent) transpose: jit, residual jets (K+1)+W+(N,) -> variation gradient
        rt = tuple(jnp.asarray(np.random.randn(4, 2, N)) for N in STRUCT[0])  # K+1=4, W=(2,)
        self.assert_jit_jax(
            lambda rr, w, p, b: pd.probe_tangent_derivatives_transpose(rr, w, p, b, 3, sum_over_probes=True),
            rt, list(self.ww), list(self.zz), self.base.data)
        # transpose with a base/core stack C=(2,): residual jets (K+1)+W+C+(N,), both sum_over_probes
        baseC = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=(2,)).to_jax())[0].data
        rtC = tuple(jnp.asarray(np.random.randn(4, 2, 2, N)) for N in STRUCT[0])  # K+1=4, W=(2,), C=(2,)
        for sop in (True, False):
            self.assert_jit_jax(
                lambda rr, w, p, b: pd.probe_tangent_derivatives_transpose(rr, w, p, b, 3, sum_over_probes=sop),
                rtC, list(self.ww), list(self.zz), baseC)
        # K-stacked Riemannian forward (exercises the order-threaded 3-block W/K/C contractions under jit)
        self.assert_jit_jax(
            lambda var, b, w, p: pd.probe_tangent_derivatives(w, p, var, b, 3),
            self.v_vstack.variations.data, self.base.data, list(self.ww), list(self.zz))
        # K-stacked transpose (the order-threaded 3-block ADJOINT contractions): residual (order+1)+W+K+C
        rtK = tuple(jnp.asarray(np.random.randn(4, 2, 3, N)) for N in STRUCT[0])  # order+1=4, W=(2,), K=(3,)
        for sop in (True, False):
            self.assert_jit_jax(
                lambda rr, w, p, b: pd.probe_tangent_derivatives_transpose(rr, w, p, b, 3, sum_over_probes=sop),
                rtK, list(self.ww), list(self.zz), self.base.data)
        # apply derivatives: Euclidean (W+C), Riemannian single, Riemannian K-stacked
        self.assert_jit_jax(
            lambda cc, w, p: pd.apply_derivatives_t3(w, p, cc, 3),
            self.x.data, list(self.ww), list(self.zz))
        self.assert_jit_jax(
            lambda var, b, w, p: pd.apply_tangent_derivatives(w, p, var, b, 3),
            self.var.data, self.base.data, list(self.ww), list(self.zz))
        self.assert_jit_jax(
            lambda var, b, w, p: pd.apply_tangent_derivatives(w, p, var, b, 3),
            self.v_vstack.variations.data, self.base.data, list(self.ww), list(self.zz))
        # entries derivatives: Euclidean and Riemannian (index a dynamic gather; general perturbation P)
        idx = jnp.array([[1, 2], [2, 3], [3, 4]])              # (d,) + W, W=(2,)
        self.assert_jit_jax(
            lambda cc, ix, p: pd.entries_derivatives_t3(ix, p, cc, 3),
            self.x.data, idx, list(self.zz))
        self.assert_jit_jax(
            lambda var, b, ix, p: pd.entries_tangent_derivatives(ix, p, var, b, 3),
            self.var.data, self.base.data, idx, list(self.zz))
        # the new order-threaded 3-block contractions directly (K=(3,), C=())
        sig = jnp.ones((4, 2, 3, 5)); Qc = jnp.ones((5, 4, 6)); xij2 = jnp.ones((2, 2, 4))
        self.assert_jit_jax(lambda a, b, c, e: contractions.trs_rWKCa_Caib_sWCi_to_tWKCb(a, b, c, e),
                            trs[:, :, :2], sig, Qc, xij2)
        dGc = jnp.ones((3, 5, 4, 6)); muc = jnp.ones((4, 2, 5))
        self.assert_jit_jax(lambda a, b, c, e: contractions.trs_rWCa_KCaib_sWCi_to_tWKCb(a, b, c, e, 0),
                            trs[:, :, :2], muc, dGc, xij2)
        # apply/entries derivative transpose (adjoint-state seeded sweep): residual jet c is a scalar
        # (order+1)+W+K+C. Single tangent (K=()) and K-stacked; both sum_over_probes; entries gathers idx.
        ca = jnp.asarray(np.random.randn(4, 2))        # order+1=4, W=(2,), K=(), C=()
        caK = jnp.asarray(np.random.randn(4, 2, 3))    # K=(3,)
        for sop in (True, False):
            self.assert_jit_jax(
                lambda cc, w, p, b: pd.apply_tangent_derivatives_transpose(cc, w, p, b, 3, sum_over_probes=sop),
                ca, list(self.ww), list(self.zz), self.base.data)
            self.assert_jit_jax(
                lambda cc, w, p, b: pd.apply_tangent_derivatives_transpose(cc, w, p, b, 3, sum_over_probes=sop),
                caK, list(self.ww), list(self.zz), self.base.data)
            self.assert_jit_jax(
                lambda cc, ix, p, b: pd.entries_tangent_derivatives_transpose(cc, ix, p, b, 3, sum_over_probes=sop),
                ca, idx, list(self.zz), self.base.data)
        # corewise derivative transposes (P,Q,O->G substitution): gradient w.r.t. the base's own cores
        self.assert_jit_jax(
            lambda rr, w, p, cp: pd.probe_corewise_derivatives_transpose(rr, w, p, cp, 3, sum_over_probes=True),
            rt, list(self.ww), list(self.zz), self.x.data)
        self.assert_jit_jax(
            lambda cc, w, p, cp: pd.apply_corewise_derivatives_transpose(cc, w, p, cp, 3, sum_over_probes=True),
            ca, list(self.ww), list(self.zz), self.x.data)
        self.assert_jit_jax(
            lambda cc, ix, p, cp: pd.entries_corewise_derivatives_transpose(cc, ix, p, cp, 3, sum_over_probes=True),
            ca, idx, list(self.zz), self.x.data)

    # ---------------------------------------------------- jit bucket: backend functions
    def test_jit_backend(self):
        # one custom contraction (contractions.py)
        FGa = jnp.ones((2, 3)); Gaib = jnp.ones((3, 4, 5)); FGi = jnp.ones((2, 4))
        self.assert_jit_jax(lambda a, b, c: contractions.WCa_Caib_WCi_to_WCb(a, b, c), FGa, Gaib, FGi)
        # orthogonal_representations (orthogonal_representations.py) -> returns (T3Basis, T3Variations)
        self.assert_jit_jax(lambda a: bvf.t3_orthogonal_representations(a), self.x)
        # tangent backend (tangent_operations.py)
        self.assert_jit_jax(lambda b, v: tops.tangent_to_dense(b, v), self.base.data, self.var.data)
        # contraction-only dense projection (no SVD -> static shapes -> jit-able)
        dense = jnp.asarray(np.random.randn(*STRUCT[0]))
        self.assert_jit_jax(lambda b, z: tops.project_dense_onto_tangent_space(b, z), self.base.data, dense)
        # residual / checker backends -> jax scalar (raw-np dispatch fix)
        self.assert_jit_jax(lambda b: orth_reps.basis_orthogonality_residual(b), self.base.data)
        self.assert_jit_jax(lambda b: orth_reps.basis_consistency_residual(b), self.base.data)
        self.assert_jit_jax(lambda b, v: tops.gauge_residual(b, v), self.base.data, self.var.data)

    # ---------------------------------------------------- jit bucket: Gauss-Newton fitting (fitting.py)
    def test_jit_fitting(self):
        # the geometry-generic GN model, every (kind x geometry): cached sweep + base fold in as closure
        # constants; the trial tangent pp is the traced input (basis survives jit as aux). evaluate exercises
        # the kind's sumsq reducer; gn_hessian exercises forward + transpose + geometry.project.
        index = jnp.array([[1, 2], [2, 3], [3, 0]])              # (d,)+W, W=(2,)
        probe_r = tuple(jnp.ones((2, N)) for N in STRUCT[0])     # d probe residual vectors, W=(2,)
        for geom in (t3m.MANIFOLD, t3m.COREWISE):                # corewise: NO Π; manifold: gauged
            models = [fitting.apply_model(geom, self.x, self.ww, jnp.ones(2)),
                      fitting.entries_model(geom, self.x, index, jnp.ones(2)),
                      fitting.probe_model(geom, self.x, self.ww, probe_r)]
            for model in models:
                _ = model.gradient; _ = model.objective_value   # warm the caches -> concrete jax constants
                p = geom.randn(model.base)                       # a tangent at the model's base (jax)
                self.assert_jit_jax(lambda pp: model.gn_hessian(pp), p)    # H p, returns a T3Tangent
                self.assert_jit_jax(lambda pp: model.jacobian(pp), p)      # J p (forward), sample-space
                self.assert_jit_jax(lambda pp: model.gn_quadratic(pp), p)  # pᵀHp = ‖Jp‖², a jax scalar
                self.assert_jit_jax(lambda pp: model.evaluate(pp), p)      # m(p), returns a jax scalar

    def test_jit_geometry_as_arg(self):
        # the stateless geometry singletons are zero-leaf pytrees -> pass as ordinary traced args
        self.assert_jit_jax(lambda gm, t: gm.norm(gm.project(t)), t3m.MANIFOLD, self.w)
        self.assert_jit_jax(lambda gm, t: gm.norm(gm.project(t)), t3m.COREWISE, self.w)

    def test_jit_optimizer_wholestep(self):
        # Pattern 1 (the per-step jit): jit the WHOLE step, X in / X_new out, model built INSIDE. The base
        # is recomputed from the traced X each step, so it is traced -- the step COMPILES ONCE and is
        # reused across steps even though the base changes every step (no model registration needed).
        ww, b = self.ww, jnp.ones(2)
        traces = [0]
        @jax.jit
        def step(X):
            traces[0] += 1                                       # +1 per TRACE (compile), not per call
            r = X.apply(ww) - b
            model = fitting.apply_model(t3m.MANIFOLD, X, ww, r)
            g = model.gradient
            alpha = g.corewise_inner(g) / model.gn_quadratic(g)           # Cauchy step (one forward; no H assembly)
            return t3m.MANIFOLD.retract((-alpha) * g)
        X = self.x
        for _ in range(3):
            X = step(X)
        self.assertEqual(traces[0], 1, "whole-step jit recompiled -- base should be traced-internal, not aux")
        self._leaves_all_jax(X)

    # ---------------------------------------------------- jit bucket: UniformTuckerTensorTrain
    def test_jit_uniform(self):
        # Slice 7: a jitted uniform op must (a) dispatch to pure jax on the supercores and (b) -- for ops
        # that RETURN a ut3 -- keep the masks CONCRETE (host numpy), never leaking a tracer into aux_data.
        # Masks ride as static aux_data (closed over via the frontend pytree), traced only via supercores.
        ux, uy = self.ux, self.uy

        # scalar / array outputs (no ut3): host-int shape/rank extraction must work under jit
        self.assert_jit_uniform(lambda u: u.to_dense(), ux)
        self.assert_jit_uniform(lambda a, b: a.inner(b), ux, uy)
        self.assert_jit_uniform(lambda a, b: a.inner(b, use_orthogonalization=False), ux, uy)
        self.assert_jit_uniform(lambda u: u.norm(), ux)
        self.assert_jit_uniform(lambda u: u.norm(use_orthogonalization=False), ux)
        self.assert_jit_uniform(lambda u, *v: u.apply(v), ux, *self.uvecs)
        self.assert_jit_uniform(lambda u, i: u.entries(i), ux, self.uidx)
        self.assert_jit_uniform(lambda u, *w: u.probe(w), ux, *self.uww)
        self.assert_jit_uniform(lambda u: u.sum(), ux)

        # ops returning a ut3: ALSO check the masks stay concrete (the tracer-leak failure mode)
        self.assert_jit_uniform(lambda u: u.sum_stack(), ux, returns_ut3=True)
        self.assert_jit_uniform(lambda u: u.reverse(), ux, returns_ut3=True)
        self.assert_jit_uniform(lambda u: u.squash_tails(), ux, returns_ut3=True)
        self.assert_jit_uniform(lambda a, b: a + b, ux, uy, returns_ut3=True)
        self.assert_jit_uniform(lambda u: 2.5 * u, ux, returns_ut3=True)
        for m in ('down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores',
                  'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores'):
            self.assert_jit_uniform(lambda u, mm=m: getattr(u, mm)(), ux, returns_ut3=True)
        # t3svd at FIXED max-ranks -> static shapes -> jit-able; output ut3 masks stay concrete
        self.assert_jit_uniform(
            lambda u: u.t3svd(max_tucker_ranks=2, max_tt_ranks=2)[0], ux, returns_ut3=True)

        # array-in constructor under jit: a jax TuckerTensorTrain in -> jax supercores out, masks stay
        # concrete. (from_t3 is the array-taking uniform constructor; from_canonical / from_tensor_train
        # were removed as ambiguous ragged round-trips. Pure zeros/ones/randn have no array input ->
        # jax-out only, covered by test_jax_out_uniform_ctors.)
        self.assert_jit_uniform(lambda x: ut3.UniformTuckerTensorTrain.from_t3(x), self.x_np.to_jax(), returns_ut3=True)

    # ---------------------------------------------------- jit bucket: ragged <-> uniform bv converters (2c-A)
    def test_jit_cross_layer_bv_converters(self):
        # from_t3basis / from_t3variations: jax ragged cores in -> jax supercores out, masks stay host
        # (same pad+stack-via-xnp / np-host-mask machinery as from_t3). to_t3basis / to_t3variations:
        # argwhere on host masks indexes jax supercores -> jax ragged out.
        import t3toolbox.uniform_basis_variations_format as ubv
        self.assert_jit_uniform(lambda b: ubv.UT3Basis.from_t3basis(b), self.base, returns_ut3=True)
        self.assert_jit_uniform(lambda v: ubv.UT3Variations.from_t3variations(v), self.var, returns_ut3=True)
        UB = ubv.UT3Basis.from_t3basis(self.base)
        UV = ubv.UT3Variations.from_t3variations(self.var)
        self.assert_jit_jax(lambda b: b.to_t3basis(), UB)
        self.assert_jit_jax(lambda v: v.to_t3variations(), UV)
        # 2c-B base-point conversions: to_ut3 returns a ut3 (masks must stay concrete); to_dense -> array;
        # from_ut3 runs the orthogonal-representation sweep under jit (its frame masks stay concrete too).
        self.assert_jit_uniform(lambda b: b.to_ut3(), UB, returns_ut3=True)
        self.assert_jit_uniform(lambda b: b.to_dense(), UB)
        self.assert_jit_uniform(lambda u: ubv.UT3Basis.from_ut3(u).to_ut3(), self.ux, returns_ut3=True)

    # ------------------------------------------- performance contract: value-hashed masks => no recompile
    def test_mask_rebuild_does_not_recompile(self):
        # A uniform object's masks ride as jax aux_data, so their __hash__/__eq__ are part of the jit cache
        # key. Rebuilding the masks (a fresh-but-array-identical holder -- e.g. re-orthogonalizing the frame
        # every optimization step) must NOT recompile, or the loop pays a recompile every iteration. The
        # ValueHashedMasks mixin makes the key reflect rank STRUCTURE, not object identity. (The Python body
        # of a jitted fn runs once per TRACE, so the counter == number of compilations.)
        import t3toolbox.uniform_basis_variations_format as ubv

        n_plain = [0]
        @jax.jit
        def fn_plain(u):
            n_plain[0] += 1
            return u.norm()
        for _ in range(4):
            fn_plain(ut3.UniformTuckerTensorTrain.from_t3(self.x_np).to_jax())   # fresh UT3Masks each call, identical structure
        self.assertEqual(n_plain[0], 1, 'plain UT3 recompiled on mask rebuild (mask hash/eq not value-based)')

        # same for the orthogonal frame's UT3BasisMasks (the optimization-loop case)
        d, N, nU, nD, rL, rR = 3, 6, 4, 5, 3, 2
        pm = lambda r, p: np.arange(p) < np.asarray(r)[..., None]
        def make_frame():
            up    = jnp.asarray(np.random.randn(d, nU, N))
            down  = jnp.asarray(np.random.randn(d, rL, nD, rR))
            left  = jnp.asarray(np.random.randn(d, rL, nU, rL))
            right = jnp.asarray(np.random.randn(d, rR, nU, rR))
            masks = ubv.UT3BasisMasks(pm([2,3,4],nU), pm([3,4,5],nD), pm([1,2,3,1],rL), pm([1,2,2,1],rR))
            return ubv.UT3Basis(up, down, left, right, (4,5,6), masks)
        n_frame = [0]
        @jax.jit
        def fn_frame(B):
            n_frame[0] += 1
            return sum(jnp.sum(c) for c in B.data[:4])
        for _ in range(4):
            fn_frame(make_frame())                        # fresh UT3BasisMasks each call, identical structure
        self.assertEqual(n_frame[0], 1, 'UT3Basis recompiled on frame rebuild (mask hash/eq not value-based)')

    # ---------------------------------------------------- jax-out bucket: pure uniform constructors
    def test_jax_out_uniform_ctors(self):
        # pure constructors (no array input) take use_jax=True -> jax supercores out, masks numpy (host).
        for ctor in (
            lambda: ut3.UniformTuckerTensorTrain.zeros((4, 5, 6), (2, 2, 2), (1, 2, 2, 1), use_jax=True),
            lambda: ut3.UniformTuckerTensorTrain.ones((4, 5, 6), use_jax=True),
            lambda: ut3.UniformTuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1), use_jax=True),
        ):
            u = ctor()
            self._leaves_all_jax(u)        # supercores are jax
            self.assert_concrete_masks(u)  # masks stay host numpy

    # ---------------------------------------------------- output-check bucket: dynamic-shape ops
    def test_dynamic_shape_dispatch(self):
        # rtol/atol truncation chooses ranks from the data -> dynamic shapes -> cannot jit.
        self.assert_eager_jax(lambda a: a.t3svd(rtol=0.05), self.x)
        A = jnp.array(np.random.randn(8, 9))
        self.assert_eager_jax(lambda m: linalg.truncated_svd(m, rtol=0.05), A)
        # MANIFOLD.project_ambient (dense grad, 'contraction'/'t3svd'): t3svd_dense picks ranks from the data
        # -> dynamic shapes. Dispatch is pushed down (no top-level jax check), so the output is jax
        # whenever ANY input is jax. jax dense + jax basis:
        dense = jnp.array(np.random.randn(*STRUCT[0]))
        self.assert_eager_jax(lambda z: t3m.MANIFOLD.project_ambient(self.base, z), dense)
        self.assert_eager_jax(lambda z: t3m.MANIFOLD.project_ambient(self.base, z), dense)
        # jax dense + NUMPY basis: the COMPUTED variations must be jax (any input jax -> jax); the old
        # code coerced the dense down to the basis's numpy here -- the regression this fix prevents. With
        # basis-as-leaf the tangent also carries the (numpy) basis as leaves, so check the variations only.
        base_np = self.base.to_numpy()
        self._leaves_all_jax(t3m.MANIFOLD.project_ambient(base_np, dense).variations)

    # ---------------------------------------------------- numerical smoke tests (jax == numpy)
    def test_jax_matches_numpy_smoke(self):
        # A few complex ops: jax must agree with numpy (guards subtle backend divergence).
        base_np, var_np = bvf.t3_orthogonal_representations(self.x_np)
        v_np = t3m.T3Tangent(base_np, var_np)
        ww_np = tuple(np.asarray(w) for w in self.ww)

        def close(a, b):
            a, b = np.asarray(a), np.asarray(b)
            self.assertLessEqual(norm(a - b), tol * max(1.0, norm(b)))

        x_other_np = self.x_other.to_numpy()
        for m in ('inplace_fused', 'swap'):                                   # t3m: jax == numpy (max-rank)
            close(self.x.t3m(self.x_other, method=m, max_tucker_ranks=2, max_tt_ranks=2, **(
                      {'oversample': 2} if m == 'swap' else {})).to_dense(),
                  self.x_np.t3m(x_other_np, method=m, max_tucker_ranks=2, max_tt_ranks=2, **(
                      {'oversample': 2} if m == 'swap' else {})).to_dense())
        close(self.x.to_dense(), self.x_np.to_dense())                       # TTT.to_dense
        close(self.v.to_dense(), v_np.to_dense())                            # Tangent.to_dense
        close(t3m.MANIFOLD.retract(self.v).to_dense(), t3m.MANIFOLD.retract(v_np).to_dense())        # retract (fixed-rank t3svd)
        w_np = t3m.T3Tangent(base_np, bvf.T3Variations(
            tuple(np.asarray(c) for c in self.w.variations.tucker_variations),
            tuple(np.asarray(c) for c in self.w.variations.tt_variations)))
        close(self.v.corewise_inner(self.w), v_np.corewise_inner(w_np))      # inner (binary, shared base)
        for a, b in zip(self.v.probe(self.ww), v_np.probe(ww_np)):           # probe
            close(a, b)


if __name__ == "__main__":
    unittest.main()
