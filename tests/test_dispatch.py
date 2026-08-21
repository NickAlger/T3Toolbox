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
import dataclasses as dc
import numpy as np
import unittest

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.fv_operations as fv_operations
import t3toolbox.corewise as _cw
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.common as common
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.sampling_derivatives as pd
import t3toolbox.backend.tv_operations as tops
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.probing as probing
import t3toolbox.backend.sharing as bsharing
import t3toolbox.shared_geometry as sgm
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
        cls.frame, cls.var = bvf.t3_orthogonal_representations(cls.x)
        cls.v = t3m.T3Tangent(cls.frame, cls.var)
        cls.w = t3m.COREWISE.randn(cls.frame)
        cls.v_vstack = t3m.COREWISE.randn(cls.frame, stack_shape=(3,))  # K=(3,)
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
        # grouped t3svd + grouped adjustment (sharing static; fixed caps -> static shapes; the
        # safe-mode tied check skips under the trace)
        xt = t3.TuckerTensorTrain((jnp.asarray(np.random.randn(3, 5)),) * 2,
                                  (jnp.asarray(np.random.randn(1, 3, 2)),
                                   jnp.asarray(np.random.randn(2, 3, 1))))
        self.assert_jit_jax(
            lambda a: a.t3svd(max_tucker_ranks=2, max_tt_ranks=2, sharing=(0, 0)), xt)
        self.assert_jit_jax(
            lambda a: a.t3svd(sharing=(0, 0))[0].rank_adjustment_sweep('right_to_left',
                                                                       sharing=(0, 0)), xt)
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
        # weighted layer: absorb edge weights into the cores (t3_absorb_weights); T3Weights is a pytree
        Wj = t3.T3Weights(tuple(jnp.ones(n) for n in self.x.tucker_ranks),
                          tuple(jnp.ones(r) for r in self.x.tt_ranks))
        self.assert_jit_jax(lambda a, w: t3.t3_absorb_weights(a, w), self.x, Wj)
        self.assert_jit_jax(lambda a, w: t3.t3_weighted_norm(a, w), self.x, Wj)
        self.assert_jit_jax(lambda a, w: t3.t3_weighted_inner(a, w, a, w), self.x, Wj)
        self.assert_jit_jax(lambda w1, w2: w1.concatenate(w2), Wj, Wj)  # ranks add
        self.assert_jit_jax(lambda w1, w2: w1.kronecker(w2), Wj, Wj)    # ranks multiply
        self.assert_jit_jax(lambda w: bvf.T3FrameWeights.from_t3weights(w), Wj)  # T3Weights -> tangent metric
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
        # corewise (non-manifold) adjoints: gradient w.r.t. the cores (frame x passed as a traced pytree)
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
        frame = self.frame  # close over the fixed frame (aux_data); never a traced arg
        self.assert_jit_jax(lambda a: a.to_dense(), self.v)
        self.assert_jit_jax(lambda a: a.to_t3(), self.v)
        self.assert_jit_jax(lambda a: t3m.MANIFOLD.retract(a), self.v)
        self.assert_jit_jax(lambda a, b: a.corewise_inner(b), self.v, self.w)   # binary op; same-frame guard skips under the trace
        self.assert_jit_jax(lambda a: a.corewise_norm(), self.v)
        # weighted layer: T3FrameWeights metric on the variations (absorb into V,H -> corewise)
        Vv, Hv = self.v.variations.data
        Wf = bvf.T3FrameWeights(tuple(jnp.ones(H.shape[-2]) for H in Hv), tuple(jnp.ones(V.shape[-2]) for V in Vv),
                                tuple(jnp.ones(H.shape[-3]) for H in Hv), tuple(jnp.ones(H.shape[-1]) for H in Hv))
        self.assert_jit_jax(lambda a, w: a.weighted_norm(w), self.v, Wf)
        self.assert_jit_jax(lambda a, b, w: a.weighted_inner(b, w), self.v, self.w, Wf)
        self.assert_jit_jax(lambda a, w: a.absorb_weights(w), self.v, Wf)  # -> weighted T3Variations
        self.assert_jit_jax(lambda w1, w2: w1.concatenate(w2), Wf, Wf)
        self.assert_jit_jax(lambda w1, w2: w1.kronecker(w2), Wf, Wf)
        self.assert_jit_jax(lambda a, b: a + b, self.v, self.w)
        self.assert_jit_jax(lambda a, b: a - b, self.v, self.w)
        self.assert_jit_jax(lambda a: 2.5 * a, self.v)
        self.assert_jit_jax(lambda a: t3m.MANIFOLD.project(a), self.v)
        self.assert_jit_jax(lambda a: t3m.MANIFOLD.project_oblique(a), self.v)
        self.assert_jit_jax(lambda xx: t3m.MANIFOLD.project_ambient(frame, xx), self.x_other)
        self.assert_jit_jax(lambda a, w: a.probe(w), self.v, self.ww)
        self.assert_jit_jax(lambda a, w: a.probe(w), self.v_vstack, self.ww)  # 3-group (W,K,C) probe
        self.assert_jit_jax(lambda z, w: t3m.T3Tangent.probe_transpose(z, w, frame), self.zz, self.ww)
        # K-stacked residuals (W+K+C) -> 3-group transpose assemble, both sum modes
        self.assert_jit_jax(
            lambda z, w: t3m.T3Tangent.probe_transpose(z, w, frame, sum_over_probes=True), self.zz_vstack, self.ww)
        self.assert_jit_jax(
            lambda z, w: t3m.T3Tangent.probe_transpose(z, w, frame), self.zz_vstack, self.ww)
        self.assert_jit_jax(lambda a, w: a.apply(w), self.v_vstack, self.ww)              # tangent apply
        self.assert_jit_jax(lambda a, i: a.entries(i), self.v_vstack, jnp.array([1, 2, 3]))  # tangent entries
        # tangent adjoints: c shape W (+C); both sum modes
        self.assert_jit_jax(
            lambda cc, w: t3m.T3Tangent.apply_transpose(cc, w, frame, sum_over_probes=True), jnp.ones(2), self.ww)
        self.assert_jit_jax(
            lambda cc, w: t3m.T3Tangent.apply_transpose(cc, w, frame), jnp.ones(2), self.ww)  # keep W
        self.assert_jit_jax(
            lambda cc, i: t3m.T3Tangent.entries_transpose(cc, i, frame, sum_over_probes=True),
            jnp.ones(()), jnp.array([1, 2, 3]))

    # ---------------------------------------------------- jit bucket: symmetric probe derivatives
    def test_jit_probe_derivatives(self):
        # paired (X, P) sample stack W=(2,); order static; all-orders jet output must be all-jax.
        self.assert_jit_jax(
            lambda cc, w, p: pd.t3_probe_derivatives(w, p, cc, 3),
            self.x.data, list(self.ww), list(self.zz))
        # with a frame/core stack C=(2,) too: output (K+1) + W + C + (N,)
        xc = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=(2,)).to_jax()
        self.assert_jit_jax(
            lambda cc, w, p: pd.t3_probe_derivatives(w, p, cc, 3),
            xc.data, list(self.ww), list(self.zz))
        # the grouped-einsum interpreter directly: ndim solve + expansion at trace time (static
        # string/lens); a 4-operand trs combine, an order-broadcast lift, and a summed-W assemble
        trs = pd.binomial_combine_tensor(3)
        mu = jnp.ones((4, 2, 5)); G = jnp.ones((5, 4, 6)); xij = jnp.ones((2, 2, 4)); nu = jnp.ones((4, 2, 6))
        self.assert_jit_jax(lambda a, b, c, e: contractions.contract('trs,rWCa,Caib,sWCi->tWCb', a, b, c, e),
                            trs[:, :, :2], mu, G, xij)
        self.assert_jit_jax(lambda a, b, c, e: contractions.contract('trs,rWCa,Caib,sWCb->tWCi', a, b, c, e),
                            trs, mu, G, nu)
        eta = jnp.ones((4, 2, 4)); U = jnp.ones((4, 7))
        self.assert_jit_jax(lambda a, b: contractions.contract('tWCi,Cio->tWCo', a, b), eta, U)
        self.assert_jit_jax(lambda a, b: contractions.contract('WCo,WCa->Cao', a, b, len_W=1),
                            jnp.ones((2, 5)), jnp.ones((2, 6)))
        # Riemannian (tangent) forward derivatives: jit, frame + variation sweeps, all-orders
        self.assert_jit_jax(
            lambda var, b, w, p: pd.tv_probe_derivatives(w, p, var, b, 3),
            self.var.data, self.frame.data, list(self.ww), list(self.zz))
        # Riemannian (tangent) transpose: jit, residual jets (K+1)+W+(N,) -> variation gradient
        rt = tuple(jnp.asarray(np.random.randn(4, 2, N)) for N in STRUCT[0])  # K+1=4, W=(2,)
        self.assert_jit_jax(
            lambda rr, w, p, b: pd.tv_probe_derivatives_transpose(rr, w, p, b, 3, sum_over_probes=True),
            rt, list(self.ww), list(self.zz), self.frame.data)
        # transpose with a frame/core stack C=(2,): residual jets (K+1)+W+C+(N,), both sum_over_probes
        frameC = bvf.t3_orthogonal_representations(
            t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=(2,)).to_jax())[0].data
        rtC = tuple(jnp.asarray(np.random.randn(4, 2, 2, N)) for N in STRUCT[0])  # K+1=4, W=(2,), C=(2,)
        for sop in (True, False):
            self.assert_jit_jax(
                lambda rr, w, p, b: pd.tv_probe_derivatives_transpose(rr, w, p, b, 3, sum_over_probes=sop),
                rtC, list(self.ww), list(self.zz), frameC)
        # K-stacked Riemannian forward (exercises the order-threaded 3-block W/K/C contractions under jit)
        self.assert_jit_jax(
            lambda var, b, w, p: pd.tv_probe_derivatives(w, p, var, b, 3),
            self.v_vstack.variations.data, self.frame.data, list(self.ww), list(self.zz))
        # K-stacked transpose (the order-threaded 3-block ADJOINT contractions): residual (order+1)+W+K+C
        rtK = tuple(jnp.asarray(np.random.randn(4, 2, 3, N)) for N in STRUCT[0])  # order+1=4, W=(2,), K=(3,)
        for sop in (True, False):
            self.assert_jit_jax(
                lambda rr, w, p, b: pd.tv_probe_derivatives_transpose(rr, w, p, b, 3, sum_over_probes=sop),
                rtK, list(self.ww), list(self.zz), self.frame.data)
        # apply derivatives: Euclidean (W+C), Riemannian single, Riemannian K-stacked
        self.assert_jit_jax(
            lambda cc, w, p: pd.t3_apply_derivatives(w, p, cc, 3),
            self.x.data, list(self.ww), list(self.zz))
        self.assert_jit_jax(
            lambda var, b, w, p: pd.tv_apply_derivatives(w, p, var, b, 3),
            self.var.data, self.frame.data, list(self.ww), list(self.zz))
        self.assert_jit_jax(
            lambda var, b, w, p: pd.tv_apply_derivatives(w, p, var, b, 3),
            self.v_vstack.variations.data, self.frame.data, list(self.ww), list(self.zz))
        # entries derivatives: Euclidean and Riemannian (index a dynamic gather; general perturbation P)
        idx = jnp.array([[1, 2], [2, 3], [3, 4]])              # (d,) + W, W=(2,)
        self.assert_jit_jax(
            lambda cc, ix, p: pd.t3_entries_derivatives(ix, p, cc, 3),
            self.x.data, idx, list(self.zz))
        self.assert_jit_jax(
            lambda var, b, ix, p: pd.tv_entries_derivatives(ix, p, var, b, 3),
            self.var.data, self.frame.data, idx, list(self.zz))
        # the order-threaded 3-group interpreter strings directly (K=(3,), C=())
        sig = jnp.ones((4, 2, 3, 5)); Qc = jnp.ones((5, 4, 6)); xij2 = jnp.ones((2, 2, 4))
        self.assert_jit_jax(
            lambda a, b, c, e: contractions.contract('trs,rWKCa,Caib,sWCi->tWKCb', a, b, c, e),
            trs[:, :, :2], sig, Qc, xij2)
        dGc = jnp.ones((3, 5, 4, 6)); muc = jnp.ones((4, 2, 5))
        self.assert_jit_jax(
            lambda a, b, c, e: contractions.contract('trs,rWCa,KCaib,sWCi->tWKCb', a, b, c, e, len_C=0),
            trs[:, :, :2], muc, dGc, xij2)
        # apply/entries derivative transpose (adjoint-state seeded sweep): residual jet c is a scalar
        # (order+1)+W+K+C. Single tangent (K=()) and K-stacked; both sum_over_probes; entries gathers idx.
        ca = jnp.asarray(np.random.randn(4, 2))        # order+1=4, W=(2,), K=(), C=()
        caK = jnp.asarray(np.random.randn(4, 2, 3))    # K=(3,)
        for sop in (True, False):
            self.assert_jit_jax(
                lambda cc, w, p, b: pd.tv_apply_derivatives_transpose(cc, w, p, b, 3, sum_over_probes=sop),
                ca, list(self.ww), list(self.zz), self.frame.data)
            self.assert_jit_jax(
                lambda cc, w, p, b: pd.tv_apply_derivatives_transpose(cc, w, p, b, 3, sum_over_probes=sop),
                caK, list(self.ww), list(self.zz), self.frame.data)
            self.assert_jit_jax(
                lambda cc, ix, p, b: pd.tv_entries_derivatives_transpose(cc, ix, p, b, 3, sum_over_probes=sop),
                ca, idx, list(self.zz), self.frame.data)
        # corewise derivative transposes (P,Q,O->G substitution): gradient w.r.t. the frame's own cores
        self.assert_jit_jax(
            lambda rr, w, p, cp: pd.t3_probe_corewise_derivatives_transpose(rr, w, p, cp, 3, sum_over_probes=True),
            rt, list(self.ww), list(self.zz), self.x.data)
        self.assert_jit_jax(
            lambda cc, w, p, cp: pd.t3_apply_corewise_derivatives_transpose(cc, w, p, cp, 3, sum_over_probes=True),
            ca, list(self.ww), list(self.zz), self.x.data)
        self.assert_jit_jax(
            lambda cc, ix, p, cp: pd.t3_entries_corewise_derivatives_transpose(cc, ix, p, cp, 3, sum_over_probes=True),
            ca, idx, list(self.zz), self.x.data)

    # ---------------------------------------------------- jit bucket: backend functions
    def test_jit_backend(self):
        # one grouped contraction through the interpreter (contractions.py)
        FGa = jnp.ones((2, 3)); Gaib = jnp.ones((3, 4, 5)); FGi = jnp.ones((2, 4))
        self.assert_jit_jax(lambda a, b, c: contractions.contract('WCa,Caib,WCi->WCb', a, b, c),
                            FGa, Gaib, FGi)
        # t3_orthogonal_representations (fv_conversions.py) -> returns (T3Frame, T3Variations)
        self.assert_jit_jax(lambda a: bvf.t3_orthogonal_representations(a), self.x)
        # tangent backend (tv_operations.py)
        self.assert_jit_jax(lambda b, v: tops.tv_to_dense(b, v), self.frame.data, self.var.data)
        # contraction-only dense projection (no SVD -> static shapes -> jit-able)
        dense = jnp.asarray(np.random.randn(*STRUCT[0]))
        self.assert_jit_jax(lambda b, z: tops.tv_project_dense_onto_tangent_space(b, z), self.frame.data, dense)
        # residual / checker backends -> jax scalar (raw-np dispatch fix)
        self.assert_jit_jax(lambda b: fv_operations.fv_frame_orthogonality_residual(b), self.frame.data)
        self.assert_jit_jax(lambda b: fv_operations.fv_frame_consistency_residual(b), self.frame.data)
        self.assert_jit_jax(lambda b, v: tops.tv_gauge_residual(b, v), self.frame.data, self.var.data)
        # sharing residual / checker / mean-repair (sharing.py); `sharing` is static, closed over
        stk = (jnp.asarray(np.random.randn(3, 6)),) * 2
        stt = (jnp.asarray(np.random.randn(1, 3, 2)), jnp.asarray(np.random.randn(2, 3, 1)))
        self.assert_jit_jax(lambda a0, a1, g0, g1:
                            bsharing.t3_sharing_residual(((a0, a1), (g0, g1)), (0, 0)),
                            stk[0], stk[1], stt[0], stt[1])
        self.assert_jit_jax(lambda a0, a1, g0, g1:
                            bsharing.t3_tucker_factors_shared(((a0, a1), (g0, g1)), (0, 0)),
                            stk[0], stk[1], stt[0], stt[1])
        self.assert_jit_jax(lambda a0, a1, g0, g1:
                            bsharing.t3_tie_tucker_factors(((a0, a1), (g0, g1)), (0, 0)),
                            stk[0], stk[1], stt[0], stt[1])
        # the weights-compatibility residual/checker (weights carry no shape; labels-only groups)
        wv = (jnp.asarray(np.random.rand(3)),) * 2
        wtt = tuple(jnp.asarray(np.random.rand(k)) for k in (1, 2, 1))
        self.assert_jit_jax(lambda w0, w1: bsharing.t3_tucker_weights_sharing_residual(((w0, w1), wtt), (0, 0)),
                            wv[0], wv[1])
        self.assert_jit_jax(lambda w0, w1: bsharing.t3_tucker_weights_shared(((w0, w1), wtt), (0, 0)),
                            wv[0], wv[1])
        # the shared-frame companion (re-sweep + batched thin SVD); groups static, arrays leaves
        def _shared_frame_data(a, g0, g1):
            frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain((a, a), (g0, g1)))
            return bsharing.fv_shared_frame_data(frame.data, ((0, 1),))
        self.assert_jit_jax(_shared_frame_data, stk[0], stt[0], stt[1])
        # the tied post-pass through the threaded gauge projection (clip-pinv solve, all einsums)
        def _tied_projection(a, g0, g1, z):
            frame, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain((a, a), (g0, g1)))
            sfd = bsharing.fv_shared_frame_data(frame.data, ((0, 1),))
            return tops.tv_project_dense_onto_tangent_space(frame.data, z, shared_data=sfd)
        self.assert_jit_jax(_tied_projection, stk[0], stt[0], stt[1],
                            jnp.asarray(np.random.randn(6, 6)))

    def test_jit_shared_geometry_fitting(self):
        # a SharedGeometry model: the wrapper is value-hashed aux, the companion a leaf; the
        # matvec (tied projections on both sides) compiles clean
        xt = t3.TuckerTensorTrain((jnp.asarray(np.random.randn(2, 6)),) * 2,
                                  (jnp.asarray(np.random.randn(1, 2, 2)),
                                   jnp.asarray(np.random.randn(2, 2, 1))))
        geom = sgm.shared_manifold((0, 0))
        ww = tuple(jnp.asarray(np.random.randn(5, 6)) for _ in range(2))
        r = jnp.asarray(np.random.randn(5))
        model = fitting.apply_model(geom, xt, ww, r)
        p = geom.randn(model.frame)
        self.assert_jit_jax(lambda m, q: m.gn_hessian(q), model, p)
        self.assert_jit_jax(lambda m: m.gradient, model)

    # ---------------------------------------------------- jit bucket: Gauss-Newton fitting (fitting.py)
    def test_jit_fitting(self):
        # the geometry-generic GN model, every (kind x geometry): cached sweep + frame fold in as closure
        # constants; the trial tangent pp is the traced input (frame survives jit as aux). evaluate exercises
        # the kind's sumsq reducer; gn_hessian exercises forward + transpose + geometry.project.
        index = jnp.array([[1, 2], [2, 3], [3, 0]])              # (d,)+W, W=(2,)
        probe_r = tuple(jnp.ones((2, N)) for N in STRUCT[0])     # d probe residual vectors, W=(2,)
        for geom in (t3m.MANIFOLD, t3m.COREWISE):                # corewise: NO Π; manifold: gauged
            models = [fitting.apply_model(geom, self.x, self.ww, jnp.ones(2)),
                      fitting.entries_model(geom, self.x, index, jnp.ones(2)),
                      fitting.probe_model(geom, self.x, self.ww, probe_r)]
            for model in models:
                _ = model.gradient; _ = model.objective_value   # warm the caches -> concrete jax constants
                p = geom.randn(model.frame)                       # a tangent at the model's frame (jax)
                self.assert_jit_jax(lambda pp: model.gn_hessian(pp), p)    # H p, returns a T3Tangent
                self.assert_jit_jax(lambda pp: model.jacobian(pp), p)      # J p (forward), sample-space
                self.assert_jit_jax(lambda pp: model.gn_quadratic(pp), p)  # pᵀHp = ‖Jp‖², a jax scalar
                self.assert_jit_jax(lambda pp: model.evaluate(pp), p)      # m(p), returns a jax scalar
        # weighted probe kinds jit-clean: a per-mode probe weight (d,) and a full ω[mode,order] matrix are
        # host-numpy static -> fold in as device constants (no tracer leak through the weighted sumsq/transpose).
        d, order = len(STRUCT[0]), 2
        pp = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])
        jet_r = tuple(jnp.ones((order + 1, 2, N)) for N in STRUCT[0])
        wmodels = [fitting.probe_model(t3m.MANIFOLD, self.x, self.ww, probe_r,
                                       weight=np.linspace(0.4, 1.8, d)),
                   fitting.probe_derivatives_model(t3m.MANIFOLD, self.x, self.ww, pp, order, jet_r,
                                                   weight=np.linspace(0.3, 2.0, d * (order + 1)).reshape(d, order + 1))]
        for model in wmodels:
            _ = model.gradient; _ = model.objective_value
            p = t3m.MANIFOLD.randn(model.frame)
            self.assert_jit_jax(lambda pp_: model.gn_hessian(pp_), p)
            self.assert_jit_jax(lambda pp_: model.gn_quadratic(pp_), p)

    def test_jit_geometry_as_arg(self):
        # the stateless geometry singletons are zero-leaf pytrees -> pass as ordinary traced args
        self.assert_jit_jax(lambda gm, t: gm.norm(gm.project(t)), t3m.MANIFOLD, self.w)
        self.assert_jit_jax(lambda gm, t: gm.norm(gm.project(t)), t3m.COREWISE, self.w)

    def test_jit_optimizer_wholestep(self):
        # Pattern 1 (the per-step jit): jit the WHOLE step, X in / X_new out, model built INSIDE. The frame
        # is recomputed from the traced X each step, so it is traced -- the step COMPILES ONCE and is
        # reused across steps even though the frame changes every step (no model registration needed).
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
        self.assertEqual(traces[0], 1, "whole-step jit recompiled -- frame should be traced-internal, not aux")
        self._leaves_all_jax(X)

    def test_jit_uniform_optimizer_wholestep(self):
        # The UNIFORM mask-holding recipe (docs/uniform_backend_jit_recipe.md): the per-step optimizer kernel
        # closes over the host-numpy masks (via the uniform geometry/kind) and traces only the SUPERCORES, so
        # with changing supercores it COMPILES ONCE -- the frame masks re-derived inside local_model
        # constant-fold; no per-step recompile. One clean trace also proves the step is jit-clean (a stray np
        # on a tracer, or a tracer leaking into the host-numpy masks, would break the compile).
        import t3toolbox.backend.uniform_fitting as uf
        import t3toolbox.corewise as cw
        from t3toolbox.backend import apply as bapply
        SH, TK, TT = STRUCT
        W = 12
        x_true = t3.TuckerTensorTrain.randn(SH, TK, TT)
        ww = [jnp.asarray(np.random.randn(W, n)) for n in SH]
        data = jnp.asarray(bapply.t3_apply(x_true.data, [np.asarray(w) for w in ww]))
        ux0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(SH, TK, TT)).to_jax()
        sc = (ux0.data[0], ux0.data[1])                     # jax supercores; masks stay host numpy
        prob = uf.uniform_least_squares_problem('manifold', 'apply', ux0, ww, data)
        traces = [0]
        @jax.jit
        def step(supercores):
            traces[0] += 1                                  # +1 per TRACE (compile), not per call
            lm = prob.local_model(supercores)
            g = lm.gradient
            alpha = prob.geom.inner(g, g) / jnp.maximum(lm.gn_quadratic(g), 1e-30)   # Cauchy step
            return lm.retract(cw.corewise_scale(g, -alpha))
        for _ in range(3):
            sc = step(sc)                                   # changing supercores, fixed shape
        jax.block_until_ready(sc)
        self.assertEqual(traces[0], 1, "uniform whole-step jit recompiled -- masks must be closed-over host np")
        self._leaves_all_jax(ut3.UniformTuckerTensorTrain(sc[0], sc[1], ux0.shape, ux0.masks))

    def test_jit_uniform_gauss_newton_model(self):
        # U7b (the roll-your-own surface): a UniformGaussNewtonModel matvec `jit(lambda m, p: m.gn_hessian(p))`
        # -- model AS AN ARGUMENT -- must compile ONCE across REBUILT models of the same rank. This exercises
        # the value-hashed aux design: the model's aux is (geometry, kind_name, x0_masks, order, weight) -- all
        # value-hashable -- and the packed kind is rebuilt lazily from it (a fresh-closure kind stored as aux
        # would recompile every rebuild). One clean trace also proves the matvec is jit-clean (UT3Tangent
        # wrap/unwrap + gauge projection, no stray np on a tracer).
        import t3toolbox.uniform_manifold as ut3m
        SH, TK, TT = STRUCT
        W = 12
        ww = [np.random.randn(W, n) for n in SH]

        def build(seed):
            np.random.seed(seed)
            x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(SH, TK, TT)).to_jax()
            r = jnp.asarray(np.asarray(x.apply(ww)) - np.random.randn(W))
            return fitting.apply_model(ut3m.UNIFORM_MANIFOLD, x, ww, r)

        traces = [0]
        @jax.jit
        def Hmatvec(m, p):
            traces[0] += 1                                  # +1 per TRACE (compile), not per call
            return m.gn_hessian(p)
        for seed in (1, 2, 3):
            m = build(seed)                                 # rebuilt model (different frame supercores, same rank)
            p = ut3m.UNIFORM_MANIFOLD.randn(m.frame)
            hp = Hmatvec(m, p)
            jax.block_until_ready(hp.variations.supercores)
        self.assertEqual(traces[0], 1, "UniformGaussNewtonModel matvec recompiled -- aux must be value-hashed "
                                       "(kind rebuilt lazily, not stored as a fresh closure)")
        self._leaves_all_jax(hp)

    def _apply_problem_jax(self):
        SH, TK, TT = STRUCT
        np.random.seed(0)
        A = t3.TuckerTensorTrain.randn(SH, TK, TT)
        cores = (tuple(jnp.asarray(c) for c in A.data[0]), tuple(jnp.asarray(c) for c in A.data[1]))
        ww = [jnp.asarray(np.random.randn(40, n)) for n in SH]
        b = jnp.asarray(np.asarray(A.apply([np.asarray(w) for w in ww])))
        return bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b), cores

    def test_local_model_rebuilt_each_step_is_one_jit_cache_key(self):
        """A LocalModel is rebuilt at every Newton iteration by construction. It must flatten to the SAME
        treedef each time, or the inner CG recompiles per iteration -- which is exactly what it did before
        slices 1-3 (measured: 1 compile per Newton iteration on the probe_derivatives path).

        The mechanism is that the model's aux is (geometry, kind, n_w, regularizer) and the first two are
        now value-typed dataclasses, plus a static skeleton holding the frame's shape and rank masks --
        which must NOT be traced (`require_concrete_masks` rejects a traced mask outright)."""
        problem, cores = self._apply_problem_jax()
        traces = [0]

        @jax.jit
        def objective_of(model):
            traces[0] += 1                       # +1 per TRACE (compile), not per call
            return model.objective

        for _ in range(3):
            lm = problem.local_model(cores)      # REBUILT, as the outer loop does
            jax.block_until_ready(objective_of(lm))
        self.assertEqual(traces[0], 1, "LocalModel recompiled -- its aux must be value-stable")

        leaves, _ = jax.tree_util.tree_flatten(problem.local_model(cores))
        masks = [l for l in leaves if hasattr(l, 'dtype') and np.asarray(l).dtype == bool]
        self.assertEqual(masks, [], "rank masks must stay in the aux, never among the traced leaves")

    def test_cg_tolerance_is_not_stale(self):
        """The CG tolerance changes every Newton iteration (it is the forcing term). A `lax.while_loop`
        body caches on its own identity, so a stable body READING a changed Python value would silently
        get the cached jaxpr with the OLD value -- solving to a stale tolerance. `tol` and `maxiter` ride
        in the loop state and are traced arguments, so that is unrepresentable.

        Tightening the tolerance must therefore cost strictly more CG iterations, not the same number."""
        problem, cores = self._apply_problem_jax()
        lm = problem.local_model(cores)
        neg_g = _cw.corewise_scale(lm.gradient, -1.0)
        gnorm = float(problem.geom.inner(lm.gradient, lm.gradient)) ** 0.5

        counts = []
        for frac in (0.5, 0.1, 0.02, 0.005):                      # progressively tighter
            _p, i, _rs, _ok = bopt._cg_solve(lm, neg_g, frac * gnorm, 200, True)
            counts.append(int(i))
        self.assertEqual(counts, sorted(counts), f"iterations must grow as tol tightens; got {counts}")
        self.assertGreater(counts[-1], counts[0], f"tolerance ignored (stale): {counts}")

    def test_a_derived_kind_does_not_inherit_its_parents_cache_key(self):
        """A kind with DIFFERENT math must not share a jit cache key with the one it derives from.

        Regression test for a silent miscompile in the pre-class design. Kinds were a dataclass of
        lambdas plus a hand-maintained ``identity`` tuple, and ``dataclasses.replace`` copied that tuple
        unchanged -- so ``dc.replace(APPLY, forward=<something else>)`` compared EQUAL to ``APPLY``,
        jax reused ``APPLY``'s compiled program, and jit returned the unscaled answer while eager
        returned the scaled one (measured: 115.302888 vs 28.825722 on this fixture).

        Parameters are fields now and behaviour is methods, so the failure is unrepresentable: a variant
        is a subclass, which is a different type, which the value-based ``__eq__`` rejects up front."""
        SH, TK, TT = STRUCT

        @dc.dataclass(frozen=True, eq=False)
        class HalfApplyKind(bfit.ApplyKind):
            def forward(self, v, ww_, frame, sweep):
                return 0.5 * super().forward(v, ww_, frame, sweep)

        self.assertNotEqual(HalfApplyKind(), bfit.APPLY)
        self.assertNotEqual(hash(HalfApplyKind()), hash(bfit.APPLY))

        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn(SH, TK, TT)
        x = t3.TuckerTensorTrain(tuple(jnp.asarray(c) for c in x.data[0]),
                                 tuple(jnp.asarray(c) for c in x.data[1]))
        ww = [jnp.asarray(np.random.randn(15, n)) for n in SH]
        r = jnp.asarray(np.random.randn(15))

        quad = jax.jit(lambda m, p: m.gn_quadratic(p))
        plain = fitting.apply_model(t3m.MANIFOLD, x, ww, r)
        derived = dc.replace(plain, kind=HalfApplyKind())
        p0 = t3m.MANIFOLD.randn(plain.frame)

        q_plain = float(quad(plain, p0))
        q_derived = float(quad(derived, p0))
        self.assertAlmostEqual(q_derived, float(derived.gn_quadratic(p0)), places=10)  # jit == eager
        self.assertNotAlmostEqual(q_derived, q_plain, places=6)                        # and NOT the parent's

    def test_jit_ragged_gauss_newton_model_parameterized_kind(self):
        # The ragged twin of the test above, for the PARAMETERIZED kinds. A GaussNewtonModel carries its
        # SamplingKind as jax aux_data, and jax keys the compilation cache on the aux. APPLY/ENTRIES/PROBE
        # are module singletons, so they were always one object and always one compile -- but the derivative
        # and weighted kinds are BUILT PER MODEL, out of fresh closures, so under dataclass field equality
        # every rebuilt model was a new cache key and the documented "roll your own optimizer" loop
        # recompiled every outer step (measured: 3 traces for 3 rebuilds). Value-typed kinds fix it by
        # comparing the PARAMETERS (name, order, weight, chunk_size) instead of the lambdas.
        SH, TK, TT = STRUCT
        W, ORDER = 12, 2
        ww = [np.random.randn(W, n) for n in SH]
        pp = [np.random.randn(W, n) for n in SH]
        weight = np.array([[1.0, 0.5, 0.25]] * len(SH))

        def build(seed, wt):
            np.random.seed(seed)
            x = t3.TuckerTensorTrain.randn(SH, TK, TT)
            x = t3.TuckerTensorTrain(tuple(jnp.asarray(c) for c in x.data[0]),
                                     tuple(jnp.asarray(c) for c in x.data[1]))
            r = [jnp.asarray(np.asarray(z) - np.random.randn(*np.shape(z)))
                 for z in x.probe_derivatives(ww, pp, ORDER)]
            return fitting.probe_derivatives_model(t3m.MANIFOLD, x, ww, pp, ORDER, r, weight=wt)

        for label, wt in (('unweighted', None), ('weighted', weight)):
            with self.subTest(kind=label):
                traces = [0]
                @jax.jit
                def Hmatvec(m, p):
                    traces[0] += 1                          # +1 per TRACE (compile), not per call
                    return m.gn_hessian(p)
                for seed in (1, 2, 3):
                    m = build(seed, wt)                     # rebuilt model: fresh kind, same parameters
                    hp = Hmatvec(m, t3m.MANIFOLD.randn(m.frame))
                    jax.block_until_ready(hp.variations.tucker_variations[0])
                self.assertEqual(traces[0], 1,
                                 'GaussNewtonModel matvec recompiled on rebuild -- SamplingKind must '
                                 'compare by identity (parameters), not by its closure fields')

        # ... and a genuinely different kind MUST still get its own compile
        traces = [0]
        @jax.jit
        def Hmatvec2(m, p):
            traces[0] += 1
            return m.gn_hessian(p)
        Hmatvec2(build(1, None), t3m.MANIFOLD.randn(build(1, None).frame))
        Hmatvec2(build(1, weight), t3m.MANIFOLD.randn(build(1, weight).frame))
        self.assertEqual(traces[0], 2, 'different residual weights must not share a compilation')

    def test_jit_shared_uniform_gauss_newton_model(self):
        # The SHARED uniform model (slice 11): SharedGeometry(UNIFORM_MANIFOLD, sharing) is value-hashed
        # aux, the SF-T3 companion rides as the geometry_aux LEAF, and the tied matvec (companion-fed
        # projections on both sides) compiles ONCE across rebuilt same-rank models.
        import t3toolbox.uniform_manifold as ut3m
        import t3toolbox.shared_geometry as sgm
        SH, TK, TT = (5, 5, 4), (3, 3, 2), (1, 3, 2, 1)     # tied group {0,1}; shared-minimal
        W = 12
        ww = [np.random.randn(W, n) for n in SH]
        geom = sgm.shared(ut3m.UNIFORM_MANIFOLD, (0, 0, 1))

        def build(seed):
            np.random.seed(seed)
            xr = t3.TuckerTensorTrain.randn(SH, TK, TT)
            tk_b, tt_b = xr.data
            x = ut3.UniformTuckerTensorTrain.from_t3(
                t3.TuckerTensorTrain((tk_b[0], tk_b[0], tk_b[2]), tt_b)).to_jax()
            r = jnp.asarray(np.asarray(x.apply(ww)) - np.random.randn(W))
            return fitting.apply_model(geom, x, ww, r)

        traces = [0]
        @jax.jit
        def Hmatvec(m, p):
            traces[0] += 1
            return m.gn_hessian(p)
        for seed in (1, 2, 3):
            m = build(seed)
            p = geom.randn(m.frame)
            hp = Hmatvec(m, p)
            jax.block_until_ready(hp.variations.supercores)
        self.assertEqual(traces[0], 1, "shared UniformGaussNewtonModel matvec recompiled -- the wrapper "
                                       "must be value-hashed aux and the companion a leaf")
        self._leaves_all_jax(hp)

    def test_jit_uniform_weighted_gn_model(self):
        # A WEIGHTED UniformGaussNewtonModel (per-mode plain probe + full ω[mode,order] probe_derivatives)
        # must keep compile-once across rebuilds: the ω matrix rides in the value-hashed aux as a nested
        # tuple (a numpy-array aux would be unhashable / a fresh object each rebuild -> a recompile).
        import t3toolbox.uniform_manifold as ut3m
        SH, TK, TT = STRUCT
        W, d, order = 12, len(STRUCT[0]), 2
        ww = [np.random.randn(W, n) for n in SH]
        pp = [np.random.randn(W, n) for n in SH]
        omega_mode = np.linspace(0.4, 1.8, d)                        # per-mode (d,)
        wmat = np.linspace(0.3, 2.0, d * (order + 1)).reshape(d, order + 1)   # full ω[mode,order]

        def build_probe(seed):
            np.random.seed(seed)
            x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(SH, TK, TT)).to_jax()
            r = [jnp.asarray(np.random.randn(W, n)) for n in SH]
            return fitting.probe_model(ut3m.UNIFORM_MANIFOLD, x, ww, r, weight=omega_mode)

        def build_deriv(seed):
            np.random.seed(seed)
            x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(SH, TK, TT)).to_jax()
            r = [jnp.asarray(np.random.randn(order + 1, W, n)) for n in SH]
            return fitting.probe_derivatives_model(ut3m.UNIFORM_MANIFOLD, x, ww, pp, order, r, weight=wmat)

        for build in (build_probe, build_deriv):
            traces = [0]
            @jax.jit
            def Hmatvec(m, p):
                traces[0] += 1
                return m.gn_hessian(p)
            for seed in (1, 2, 3):
                m = build(seed)
                p = ut3m.UNIFORM_MANIFOLD.randn(m.frame)
                hp = Hmatvec(m, p)
                jax.block_until_ready(hp.variations.supercores)
            self.assertEqual(traces[0], 1, "weighted UniformGaussNewtonModel recompiled -- the ω matrix aux "
                                           "must be a hashable nested tuple, value-hashed like the unweighted model")
            self._leaves_all_jax(hp)

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

        # grouped (SF-T3) ut3svd / adjustment / residual under jit: the partition + masks are static
        # (closed over / aux), the supercores trace; output masks stay concrete; safe mode is skipped
        # under the trace (checks_active is trace-aware)
        xr = t3.TuckerTensorTrain.randn((5, 5, 4), (3, 3, 2), (1, 2, 2, 1))
        tk_r, tt_r = xr.data
        uxt = ut3.UniformTuckerTensorTrain.from_t3(
            t3.TuckerTensorTrain((tk_r[0], tk_r[0], tk_r[2]), tt_r)).to_jax()
        self.assert_jit_jax(lambda u: bsharing.ut3_sharing_residual(u.data, (0, 0, 1)), uxt)
        self.assert_jit_uniform(
            lambda u: u.t3svd(max_tucker_ranks=2, max_tt_ranks=2, sharing=(0, 0, 1))[0],
            uxt, returns_ut3=True)
        self.assert_jit_uniform(
            lambda u: u.t3svd(sharing=(0, 0, 1))[0].rank_adjustment_sweep('right_to_left',
                                                                          sharing=(0, 0, 1)),
            uxt, returns_ut3=True)

        # the uniform companion + tied projection + tied retraction (slice 10): masks/groups closed
        # over (static), the frame/variation supercores traced; the companion (a registered pytree:
        # arrays leaves, partition aux) flows through the trace
        import t3toolbox.backend.ufv_conversions as ufvc
        import t3toolbox.backend.utv_operations as utvo
        frame_u, var_u = ufvc.ut3_orthogonal_representations(uxt.data)
        groups = bsharing.validate_sharing((0, 0, 1), uxt.shape)
        f_shape, f_masks = frame_u[4], frame_u[5]
        v_shape, v_masks = var_u[2], var_u[3]

        def tied_step(up, dn, lf, rt, tkv, ttv):
            frame_d = (up, dn, lf, rt, f_shape, f_masks)
            var_d = (tkv, ttv, v_shape, v_masks)
            sfd = bsharing.ufv_shared_frame_data(frame_d, groups)
            tied = utvo.utv_orthogonal_gauge_projection(frame_d, var_d, shared_data=sfd)
            out = utvo.utv_retract(frame_d, tied, shared_data=sfd)
            return out[0], out[1]

        tkv0 = jnp.asarray(np.random.randn(*np.shape(var_u[0])))
        self.assert_jit_jax(tied_step, *frame_u[:4], tkv0, var_u[1])

        # array-in constructor under jit: a jax TuckerTensorTrain in -> jax supercores out, masks stay
        # concrete. (from_t3 is the array-taking uniform constructor; from_canonical / from_tensor_train
        # were removed as ambiguous ragged round-trips. Pure zeros/ones/randn have no array input ->
        # jax-out only, covered by test_jax_out_uniform_ctors.)
        self.assert_jit_uniform(lambda x: ut3.UniformTuckerTensorTrain.from_t3(x), self.x_np.to_jax(), returns_ut3=True)

    # ---------------------------------------------------- jit bucket: ragged <-> uniform bv converters (2c-A)
    def test_jit_cross_layer_fv_converters(self):
        # from_t3frame / from_t3variations: jax ragged cores in -> jax supercores out, masks stay host
        # (same pad+stack-via-xnp / np-host-mask machinery as from_t3). to_t3frame / to_t3variations:
        # argwhere on host masks indexes jax supercores -> jax ragged out.
        import t3toolbox.uniform_frame_variations_format as ubv
        self.assert_jit_uniform(lambda b: ubv.UT3Frame.from_t3frame(b), self.frame, returns_ut3=True)
        self.assert_jit_uniform(lambda v: ubv.UT3Variations.from_t3variations(v), self.var, returns_ut3=True)
        UB = ubv.UT3Frame.from_t3frame(self.frame)
        UV = ubv.UT3Variations.from_t3variations(self.var)
        self.assert_jit_jax(lambda b: b.to_t3frame(), UB)
        self.assert_jit_jax(lambda v: v.to_t3variations(), UV)
        # 2c-B base-point conversions: to_ut3 returns a ut3 (masks must stay concrete); to_dense -> array;
        # from_ut3 runs the orthogonal-representation sweep under jit (its frame masks stay concrete too).
        self.assert_jit_uniform(lambda b: b.to_ut3(), UB, returns_ut3=True)
        self.assert_jit_uniform(lambda b: b.to_dense(), UB)
        self.assert_jit_uniform(lambda u: ubv.UT3Frame.from_ut3(u).to_ut3(), self.ux, returns_ut3=True)
        # 2c-E: reverse (both classes) + orthogonalize -- masks reverse/rebuild on the host, stay concrete
        self.assert_jit_uniform(lambda b: b.reverse(), UB, returns_ut3=True)
        self.assert_jit_uniform(lambda v: v.reverse(), UV, returns_ut3=True)
        self.assert_jit_uniform(lambda b: b.orthogonalize(), UB, returns_ut3=True)
        # 2c-G2: uniform-native per-element checkers jit to jax (masked-Gram residual; masks stay np const)
        self.assert_jit_jax(lambda b: b.is_orthogonal(), UB)
        self.assert_jit_jax(lambda b: b.is_consistent(), UB)
        self.assert_jit_jax(lambda a, b: a.allclose(b), UB, ubv.UT3Frame.from_t3frame(self.frame))

    # ---------------------------------------------------- jit bucket: the uniform geometries (3b-5)
    def test_jit_uniform_geometry(self):
        # 3b-5: the uniform geometries jit cleanly -- supercores trace to jax, masks stay host constants, and
        # the per-element safe-mode preconditions skip under the trace (the uniform mirror of test_jit_tangent).
        import t3toolbox.uniform_manifold as ut3m
        M, C = ut3m.UNIFORM_MANIFOLD, ut3m.UNIFORM_COREWISE
        ux_np = ut3.UniformTuckerTensorTrain.from_t3(self.x_np)
        v = M.randn(M.frame(ux_np)).to_jax()                # gauged jax tangent (built on numpy then to_jax)
        cv = C.randn(C.frame(ux_np)).to_jax()
        frame, g = M.frame(ux_np).to_jax(), self.uy          # an orthogonal jax frame + a jax UniformTTT grad

        # frame returns a UT3Frame; retract a UniformTTT -- masks must stay concrete (the tracer-leak mode)
        self.assert_jit_uniform(lambda u: M.frame(u), self.ux, returns_ut3=True)
        self.assert_jit_uniform(lambda u: C.frame(u), self.ux, returns_ut3=True)
        self.assert_jit_uniform(lambda t: M.retract(t), v, returns_ut3=True)
        self.assert_jit_uniform(lambda t: C.retract(t), cv, returns_ut3=True)

        # tangent-returning ops: jit op().to_dense() (no numpy on a tracer) + the returned tangent's frame &
        # variations masks stay concrete (UT3Tangent has no .masks of its own -- check both sub-holders)
        for op in (M.project, M.project_oblique):
            self.assert_jit_jax(lambda t, o=op: o(t).to_dense(), v)
            gt = jax.jit(lambda t, o=op: o(t))(v)
            self._leaves_all_jax(gt)
            self.assert_concrete_masks(gt.frame); self.assert_concrete_masks(gt.variations)

        # scalar metrics + ambient projection + transport (all -> jax, preconditions skip under trace)
        self.assert_jit_jax(lambda a, b: M.inner(a, b), v, v)
        self.assert_jit_jax(lambda t: M.norm(t), v)
        self.assert_jit_jax(lambda a, b: C.inner(a, b), cv, cv)
        self.assert_jit_jax(lambda b, gg: M.project_ambient(b, gg).to_dense(), frame, g)
        self.assert_jit_jax(lambda t, b: M.transport(t, b).to_dense(), v, frame)

    # ---------------------------------------------------- jit bucket: uniform tangent probing (3b-6b)
    def test_jit_uniform_probing(self):
        # 3b-6b: the bare forward 𝒥 (probe / apply / entries on UT3Tangent) jits -- the supercores trace to
        # jax (through the d-prefixed WKC contractions + the scan sweeps), the masks stay host, and pack /
        # unpack slice with static shapes (the int-tuple shape). K-stacked to exercise the W/K/C blocks.
        import t3toolbox.uniform_manifold as ut3m
        frame = ut3m.UNIFORM_MANIFOLD.frame(ut3.UniformTuckerTensorTrain.from_t3(self.x_np))
        v = ut3m.UNIFORM_COREWISE.randn(frame, stack_shape=(2,)).to_jax()   # K = (2,)
        ww = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])    # W = (2,)
        idx = jnp.array([[1, 2], [2, 3], [3, 0]])                         # (d,) + W
        self.assert_jit_jax(lambda t, *w: t.probe(w), v, *ww)
        self.assert_jit_jax(lambda t, *w: t.apply(w), v, *ww)
        self.assert_jit_jax(lambda t, i: t.entries(i), v, idx)

        # 3b-6c transposes 𝒥ᵀ: residual -> UT3Tangent; jit the whole op + .to_dense() (supercores trace,
        # the gauge masks stay host constants). The frame (jax frame, C=()) is closed over; residual traces.
        bj = v.frame
        rr = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])   # probe residual, W=(2,), K=C=()
        cc = jnp.array(np.random.randn(2))                               # scalar residual, W=(2,)
        self.assert_jit_jax(
            lambda r0, r1, r2, *w: ut3m.UT3Tangent.probe_transpose((r0, r1, r2), w, bj, sum_over_probes=True).to_dense(),
            *rr, *ww)
        self.assert_jit_jax(
            lambda c, *w: ut3m.UT3Tangent.apply_transpose(c, w, bj, sum_over_probes=True).to_dense(), cc, *ww)
        self.assert_jit_jax(
            lambda c, i: ut3m.UT3Tangent.entries_transpose(c, i, bj, sum_over_probes=True).to_dense(), cc, idx)

    # ---------------------------------------------------- jit bucket: uniform corewise transposes (3b-6c)
    def test_jit_uniform_corewise_transpose(self):
        # 3b-6c: the corewise (non-manifold) sampling transposes on UniformTuckerTensorTrain (gradient w.r.t.
        # the cores, the §6.3 substitution through the tangent transpose) jit to jax supercores; masks host.
        xu = ut3.UniformTuckerTensorTrain.from_t3(self.x_np).to_jax()
        ww = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])    # W = (2,)
        idx = jnp.array([[1, 2], [2, 3], [3, 0]])
        cc = jnp.array(np.random.randn(2))
        self.assert_jit_jax(lambda u, c, *w: u.apply_corewise_transpose(c, w, sum_over_probes=True), xu, cc, *ww)
        self.assert_jit_jax(lambda u, c, i: u.entries_corewise_transpose(c, i, sum_over_probes=True), xu, cc, idx)
        self.assert_jit_jax(lambda u, *w: u.probe_corewise_transpose(w[:3], w[3:], sum_over_probes=True),
                            xu, *ww, *ww)

    # ---------------------------------------------------- jit bucket: uniform forward derivatives (3b-6'b)
    def test_jit_uniform_derivatives(self):
        # 3b-6'b: the forward derivative jets (probe/apply/entries_derivatives) jit -- the supercores trace
        # through the d-prefixed JET contractions + the scan sweeps, the binomial tensor + masks stay host
        # constants, build_input_jets vectorizes over d (no unroll), pack/unpack slice with static shapes.
        # order is a Python int (static; sets the order-axis length). Plain UT3 + K-stacked UT3Tangent.
        import t3toolbox.uniform_manifold as ut3m
        ww = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])    # W = (2,)
        pp = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])
        idx = jnp.array([[1, 2], [2, 3], [3, 0]])                         # (d,) + W

        xu = ut3.UniformTuckerTensorTrain.from_t3(self.x_np).to_jax()     # plain layer
        self.assert_jit_jax(lambda u, *w: u.probe_derivatives(w[:3], w[3:], 2), xu, *ww, *pp)
        self.assert_jit_jax(lambda u, *w: u.apply_derivatives(w[:3], w[3:], 2), xu, *ww, *pp)
        self.assert_jit_jax(lambda u, i, *w: u.entries_derivatives(i, w, 2), xu, idx, *pp)

        v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(xu), stack_shape=(2,)).to_jax()  # K=(2,)
        self.assert_jit_jax(lambda t, *w: t.probe_derivatives(w[:3], w[3:], 2), v, *ww, *pp)
        self.assert_jit_jax(lambda t, *w: t.apply_derivatives(w[:3], w[3:], 2), v, *ww, *pp)
        self.assert_jit_jax(lambda t, i, *w: t.entries_derivatives(i, w, 2), v, idx, *pp)

        # 3b-6'c transpose derivatives 𝒥ᵀ: residual jets -> UT3Tangent (jit the whole op + .to_dense()); the
        # supercores trace through the d-prefixed JET adjoint contractions + scan sweeps, gauge masks host.
        bj = v.frame                                                     # jax frame, C=()
        rj = tuple(jnp.array(np.random.randn(3, 2, N)) for N in STRUCT[0])   # probe residual jets, (order+1,W,Ni)
        cj = jnp.array(np.random.randn(3, 2))                            # scalar residual jet, (order+1,W)
        self.assert_jit_jax(
            lambda r0, r1, r2, *w: ut3m.UT3Tangent.probe_derivatives_transpose(
                (r0, r1, r2), w[:3], w[3:], bj, 2, sum_over_probes=True).to_dense(), *rj, *ww, *pp)
        self.assert_jit_jax(
            lambda c, *w: ut3m.UT3Tangent.apply_derivatives_transpose(c, w[:3], w[3:], bj, 2, sum_over_probes=True).to_dense(),
            cj, *ww, *pp)
        self.assert_jit_jax(
            lambda c, i, *w: ut3m.UT3Tangent.entries_derivatives_transpose(c, i, w, bj, 2, sum_over_probes=True).to_dense(),
            cj, idx, *pp)

        # corewise (non-manifold) derivative transposes on the plain UT3 (gradient w.r.t. cores; §6.3)
        self.assert_jit_jax(
            lambda u, c, *w: u.apply_corewise_derivatives_transpose(c, w[:3], w[3:], 2, sum_over_probes=True), xu, cj, *ww, *pp)
        self.assert_jit_jax(
            lambda u, c, i, *w: u.entries_corewise_derivatives_transpose(c, i, w, 2, sum_over_probes=True), xu, cj, idx, *pp)
        self.assert_jit_jax(
            lambda u, r0, r1, r2, *w: u.probe_corewise_derivatives_transpose((r0, r1, r2), w[:3], w[3:], 2, sum_over_probes=True),
            xu, *rj, *ww, *pp)

    # ---------------------------------------------------- stack/unstack: masks must stay host under jax
    def test_stack_unstack_keeps_masks_host(self):
        # stacking.stack infers ONE backend per call, so stacking a jax object's supercores together with
        # its host masks would promote the masks to jax. The uniform stack ops split the two calls; this
        # eager check (stack/unstack are tree ops, not jit targets) guards that masks stay host numpy.
        import t3toolbox.uniform_frame_variations_format as ubv
        xs = t3.TuckerTensorTrain.randn(*STRUCT, stack_shape=(2,)).to_jax()
        ux = ut3.UniformTuckerTensorTrain.from_t3(xs)
        ru = ut3.UniformTuckerTensorTrain.stack(ux.unstack())          # plain UT3
        self._leaves_all_jax(ru); self.assert_concrete_masks(ru)
        frame, var = ubv.ut3_orthogonal_representations(ux)             # bv frame + variations
        for r in (ubv.UT3Frame.stack(frame.unstack()), ubv.UT3Variations.stack(var.unstack())):
            self._leaves_all_jax(r); self.assert_concrete_masks(r)

    # ---------------------------------------------------- jit bucket: UT3Variations vector-space ops (2c-D)
    def test_jit_variation_linear_algebra(self):
        # corewise ops return a UT3Variations with the SAME (unchanged) mask -> masks stay concrete; the
        # same-mask precondition runs on host-static structure (no tracer branch).
        import t3toolbox.uniform_frame_variations_format as ubv
        UV = ubv.UT3Variations.from_t3variations(self.var)
        UW = ubv.UT3Variations.randn_like(UV)                          # same frame -> same mask -> addable
        self.assert_jit_uniform(lambda a, b: a + b, UV, UW, returns_ut3=True)
        self.assert_jit_uniform(lambda a, b: a - b, UV, UW, returns_ut3=True)
        self.assert_jit_uniform(lambda a: 2.5 * a, UV, returns_ut3=True)
        self.assert_jit_uniform(lambda a: -a, UV, returns_ut3=True)
        UVS = ubv.UT3Variations.from_t3variations(self.v_vstack.variations)   # stacked (K=3)
        self.assert_jit_uniform(lambda a: a.sum_stack(), UVS, returns_ut3=True)

    # ------------------------------------------- performance contract: value-hashed masks => no recompile
    def test_mask_rebuild_does_not_recompile(self):
        # A uniform object's masks ride as jax aux_data, so their __hash__/__eq__ are part of the jit cache
        # key. Rebuilding the masks (a fresh-but-array-identical holder -- e.g. re-orthogonalizing the frame
        # every optimization step) must NOT recompile, or the loop pays a recompile every iteration. The
        # ValueHashedMasks mixin makes the key reflect rank STRUCTURE, not object identity. (The Python body
        # of a jitted fn runs once per TRACE, so the counter == number of compilations.)
        import t3toolbox.uniform_frame_variations_format as ubv

        n_plain = [0]
        @jax.jit
        def fn_plain(u):
            n_plain[0] += 1
            return u.norm()
        for _ in range(4):
            fn_plain(ut3.UniformTuckerTensorTrain.from_t3(self.x_np).to_jax())   # fresh UT3Masks each call, identical structure
        self.assertEqual(n_plain[0], 1, 'plain UT3 recompiled on mask rebuild (mask hash/eq not value-based)')

        # same for the orthogonal frame's UT3FrameMasks (the optimization-loop case)
        d, N, nU, nD, rL, rR = 3, 6, 4, 5, 3, 2
        pm = lambda r, p: np.arange(p) < np.asarray(r)[..., None]
        def make_frame():
            up    = jnp.asarray(np.random.randn(d, nU, N))
            down  = jnp.asarray(np.random.randn(d, rL, nD, rR))
            left  = jnp.asarray(np.random.randn(d, rL, nU, rL))
            right = jnp.asarray(np.random.randn(d, rR, nU, rR))
            masks = ubv.UT3FrameMasks(pm([2,3,4],nU), pm([3,4,5],nD), pm([1,2,3,1],rL), pm([1,2,2,1],rR))
            return ubv.UT3Frame(up, down, left, right, (4,5,6), masks)
        n_frame = [0]
        @jax.jit
        def fn_frame(B):
            n_frame[0] += 1
            return sum(jnp.sum(c) for c in B.data[:4])
        for _ in range(4):
            fn_frame(make_frame())                        # fresh UT3FrameMasks each call, identical structure
        self.assertEqual(n_frame[0], 1, 'UT3Frame recompiled on frame rebuild (mask hash/eq not value-based)')

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
        # whenever ANY input is jax. jax dense + jax frame:
        dense = jnp.array(np.random.randn(*STRUCT[0]))
        self.assert_eager_jax(lambda z: t3m.MANIFOLD.project_ambient(self.frame, z), dense)
        self.assert_eager_jax(lambda z: t3m.MANIFOLD.project_ambient(self.frame, z), dense)
        # jax dense + NUMPY frame: the COMPUTED variations must be jax (any input jax -> jax); the old
        # code coerced the dense down to the frame's numpy here -- the regression this fix prevents. With
        # frame-as-leaf the tangent also carries the (numpy) frame as leaves, so check the variations only.
        frame_np = self.frame.to_numpy()
        self._leaves_all_jax(t3m.MANIFOLD.project_ambient(frame_np, dense).variations)

    # ---------------------------------------------------- numerical smoke tests (jax == numpy)
    def test_jax_matches_numpy_smoke(self):
        # A few complex ops: jax must agree with numpy (guards subtle backend divergence).
        frame_np, var_np = bvf.t3_orthogonal_representations(self.x_np)
        v_np = t3m.T3Tangent(frame_np, var_np)
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
        w_np = t3m.T3Tangent(frame_np, bvf.T3Variations(
            tuple(np.asarray(c) for c in self.w.variations.tucker_variations),
            tuple(np.asarray(c) for c in self.w.variations.tt_variations)))
        close(self.v.corewise_inner(self.w), v_np.corewise_inner(w_np))      # inner (binary, shared frame)
        for a, b in zip(self.v.probe(self.ww), v_np.probe(ww_np)):           # probe
            close(a, b)


if __name__ == "__main__":
    unittest.main()


class TestTracedMaskGuard(unittest.TestCase):
    """Every uniform mask chokepoint must reject a TRACED mask with the actionable error.

    Uniform masks are static structure and must be closed over, not passed among the traced jit args
    (docs/contributor/uniform_pytree_composition.md). Getting that wrong is an easy, natural mistake --
    `jax.jit(op)(obj.data)` traces every leaf, masks included -- and without the guard it surfaces as
    jax's cryptic TracerArrayConversionError from deep inside a numpy call. `require_concrete_masks`
    turns it into an error that names the fix.

    This pins the guard at every chokepoint, on all four uniform object kinds: plain, frame, variations,
    and weights. The frame/variation ones were unguarded until 2026-07-15 -- the guard lived in
    ut3_masking and the ufv layer simply never called it.
    """

    def _assert_actionable(self, name, fn, data):
        import jax
        with self.subTest(op=name):
            with self.assertRaises(ValueError) as cm:
                jax.jit(fn)(data)          # traces every leaf -> the masks become tracers
            self.assertIn('uniform masks must be concrete', str(cm.exception))

    def test_traced_masks_rejected_everywhere(self):
        if not common.jax_available:
            self.skipTest('jax not available')
        import t3toolbox.uniform_frame_variations_format as ubvf
        import t3toolbox.backend.ut3_masking as ut3_masking
        import t3toolbox.backend.ut3_conversions as ut3_conversions
        import t3toolbox.backend.ut3_operations as ut3_operations
        import t3toolbox.backend.ufv_masking as ufv_masking
        import t3toolbox.backend.ufv_conversions as ufv_conversions

        np.random.seed(0)
        x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x).to_jax()
        frame, variations = ubvf.ut3_orthogonal_representations(ux)
        uW = ut3.UT3Weights.from_t3weights(
            t3.T3Weights.from_t3svd(x), n=ux.n, r=ux.r)
        jW = ut3.UT3Weights(common.to_jax(uW.tucker_weight_supercore),
                            common.to_jax(uW.tt_weight_supercore), uW.masks)

        # plain layer
        self._assert_actionable('ut3_apply_masks', ut3_masking.ut3_apply_masks, ux.data)
        self._assert_actionable('ut3_to_dense', ut3_conversions.ut3_to_dense, ux.data)
        self._assert_actionable('ut3_to_t3', ut3_conversions.ut3_to_t3, ux.data)
        self._assert_actionable('ut3_squash_tails', ut3_operations.ut3_squash_tails, ux.data)
        # frame / variations (unguarded before 2026-07-15)
        self._assert_actionable('ufv_apply_frame_masks',
                                ufv_masking.ufv_apply_frame_masks, frame.data)
        self._assert_actionable('ufv_apply_variations_masks',
                                ufv_masking.ufv_apply_variations_masks, variations.data)
        self._assert_actionable('ut3frame_to_t3frame',
                                ufv_conversions.ut3frame_to_t3frame, frame.data)
        self._assert_actionable('ut3variations_to_t3variations',
                                ufv_conversions.ut3variations_to_t3variations, variations.data)
        # weights
        self._assert_actionable('ut3weights_to_t3weights',
                                ut3_conversions.ut3weights_to_t3weights, jW.data)
        self._assert_actionable('ut3_reciprocal_weights',
                                ut3_operations.ut3_reciprocal_weights, jW.data)

    def test_the_right_form_works(self):
        """The counterpart: close over the masks, trace only the supercores -- the documented recipe."""
        if not common.jax_available:
            self.skipTest('jax not available')
        import jax
        import t3toolbox.backend.ufv_masking as ufv_masking
        import t3toolbox.uniform_frame_variations_format as ubvf

        np.random.seed(0)
        ux = ut3.UniformTuckerTensorTrain.from_t3(
            t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))).to_jax()
        frame, _ = ubvf.ut3_orthogonal_representations(ux)
        shape, masks = frame.shape, frame.masks.data

        masked = jax.jit(lambda up, dn, lf, rt: ufv_masking.ufv_apply_frame_masks(
            (up, dn, lf, rt, shape, masks)))(*frame.supercores)
        self.assertTrue(all(common.is_jax_ndarray(a) for a in masked))
