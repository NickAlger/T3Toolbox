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
        cls.w = t3m.T3Tangent.randn(cls.base, apply_gauge_projection=False)
        cls.v_vstack = t3m.T3Tangent.randn(cls.base, stack_shape=(3,), apply_gauge_projection=False)  # K=(3,)
        cls.ww = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])  # probe stack W=(2,)
        cls.zz = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])  # W + C + (N,), C=()
        cls.zz_vstack = tuple(jnp.array(np.random.randn(2, 3, N)) for N in STRUCT[0])  # W + K + C, K=(3,)
        cls.x_other = t3.TuckerTensorTrain.randn((4, 5, 6), (3, 3, 3), (1, 2, 2, 1)).to_jax()
        # uniform fixtures: a jax UT3 (supercores jax, masks numpy/host -- slice 7) + a second one to add/inner
        cls.ux = ut3.t3_to_ut3(cls.x_np).to_jax()
        cls.uy = ut3.t3_to_ut3(t3.TuckerTensorTrain.randn(*STRUCT)).to_jax()
        cls.uvecs = tuple(jnp.array(np.random.randn(N)) for N in STRUCT[0])
        cls.uww = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])
        cls.uidx = jnp.array([1, 2, 3])
        # constructor IO inputs (jax arrays in -> jax supercores out, masks numpy)
        cls.ufactors = tuple(jnp.array(np.random.randn(2, N)) for N in STRUCT[0])  # CP, rank 2
        cls.uttc = (jnp.array(np.random.randn(1, 4, 2)), jnp.array(np.random.randn(2, 5, 2)),
                    jnp.array(np.random.randn(2, 6, 1)))                            # tensor-train cores

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
        self.assert_jit_jax(lambda a: a.retract(), self.v)
        self.assert_jit_jax(lambda a, b: a.inner(b), self.v, self.w)   # binary op, shared base via aux
        self.assert_jit_jax(lambda a: a.norm(), self.v)
        self.assert_jit_jax(lambda a, b: a + b, self.v, self.w)
        self.assert_jit_jax(lambda a, b: a - b, self.v, self.w)
        self.assert_jit_jax(lambda a: 2.5 * a, self.v)
        self.assert_jit_jax(lambda a: a.orthogonal_gauge_projection(), self.v)
        self.assert_jit_jax(lambda a: a.oblique_gauge_projection(), self.v)
        self.assert_jit_jax(lambda xx: t3m.T3Tangent.project(xx, base), self.x_other)
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
        # the new t-contractions directly (order axis t=3 leading; W=(2,), C=())
        trs = pd.binomial_combine_tensor(3)
        mu = jnp.ones((4, 2, 5)); G = jnp.ones((5, 4, 6)); xij = jnp.ones((2, 2, 4)); nu = jnp.ones((4, 2, 6))
        self.assert_jit_jax(lambda a, b, c, e: contractions.trs_rWCa_Caib_sWCi_to_tWCb(a, b, c, e),
                            trs[:, :, :2], mu, G, xij)
        self.assert_jit_jax(lambda a, b, c, e: contractions.trs_rWCa_Caib_sWCb_to_tWCi(a, b, c, e),
                            trs, mu, G, nu)
        eta = jnp.ones((4, 2, 4)); U = jnp.ones((4, 7))
        self.assert_jit_jax(lambda a, b: contractions.tWCi_Cio_to_tWCo(a, b), eta, U)

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

        # constructors taking arrays: jax factors/cores in -> jax supercores out, masks stay concrete.
        # (pure zeros/ones/randn have no array input -> jax-out only, covered by test_jax_out_uniform_ctors.)
        self.assert_jit_uniform(
            lambda f0, f1, f2: ut3.UniformTuckerTensorTrain.from_canonical((f0, f1, f2)),
            *self.ufactors, returns_ut3=True)
        self.assert_jit_uniform(
            lambda g0, g1, g2: ut3.UniformTuckerTensorTrain.from_tensor_train((g0, g1, g2)),
            *self.uttc, returns_ut3=True)

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
        # project_dense_onto_tangent / riemannian_gradient: t3svd_dense picks ranks from the data
        # -> dynamic shapes. Dispatch is pushed down (no top-level jax check), so the output is jax
        # whenever ANY input is jax. jax dense + jax basis:
        dense = jnp.array(np.random.randn(*STRUCT[0]))
        self.assert_eager_jax(lambda z: t3m.project_dense_onto_tangent(z, self.base), dense)
        self.assert_eager_jax(lambda z: t3m.riemannian_gradient(z, self.base), dense)
        # jax dense + NUMPY basis must still give jax out (any input jax -> jax); the old code
        # coerced the dense down to the basis's numpy here -- the regression this fix prevents.
        base_np = self.base.to_numpy()
        self.assert_eager_jax(lambda z: t3m.project_dense_onto_tangent(z, base_np), dense)

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
        close(self.v.retract().to_dense(), v_np.retract().to_dense())        # retract (fixed-rank t3svd)
        w_np = t3m.T3Tangent(base_np, bvf.T3Variations(
            tuple(np.asarray(c) for c in self.w.variations.tucker_variations),
            tuple(np.asarray(c) for c in self.w.variations.tt_variations)))
        close(self.v.inner(self.w), v_np.inner(w_np))                        # inner (binary, shared base)
        for a, b in zip(self.v.probe(self.ww), v_np.probe(ww_np)):           # probe
            close(a, b)


if __name__ == "__main__":
    unittest.main()
