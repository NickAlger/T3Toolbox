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
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.common as common
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.tangent_operations as tops
import t3toolbox.backend.linalg as linalg

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

    # ---------------------------------------------------- jit bucket: TuckerTensorTrain
    def test_jit_tucker_tensor_train(self):
        vecs = tuple(jnp.ones(N) for N in STRUCT[0])
        idx = jnp.array([1, 2, 3])
        self.assert_jit_jax(lambda a: a.to_dense(), self.x)
        self.assert_jit_jax(lambda a, v0, v1, v2: a.apply((v0, v1, v2)), self.x, *vecs)
        self.assert_jit_jax(lambda a, i: a.entries(i), self.x, idx)
        self.assert_jit_jax(  # t3svd with FIXED ranks -> static shapes -> jit-able
            lambda a: a.t3svd(max_tucker_ranks=(2, 2, 2), max_tt_ranks=(1, 2, 2, 1)), self.x)

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

    # ---------------------------------------------------- jit bucket: backend functions
    def test_jit_backend(self):
        # one custom contraction (contractions.py)
        FGa = jnp.ones((2, 3)); Gaib = jnp.ones((3, 4, 5)); FGi = jnp.ones((2, 4))
        self.assert_jit_jax(lambda a, b, c: contractions.WCa_Caib_WCi_to_WCb(a, b, c), FGa, Gaib, FGi)
        # orthogonal_representations (orthogonal_representations.py) -> returns (T3Basis, T3Variations)
        self.assert_jit_jax(lambda a: bvf.t3_orthogonal_representations(a), self.x)
        # tangent backend (tangent_operations.py)
        self.assert_jit_jax(lambda b, v: tops.tangent_to_dense(b, v), self.base.data, self.var.data)

    # ---------------------------------------------------- output-check bucket: dynamic-shape ops
    def test_dynamic_shape_dispatch(self):
        # rtol/atol truncation chooses ranks from the data -> dynamic shapes -> cannot jit.
        self.assert_eager_jax(lambda a: a.t3svd(rtol=0.05), self.x)
        A = jnp.array(np.random.randn(8, 9))
        self.assert_eager_jax(lambda m: linalg.truncated_svd(m, rtol=0.05), A)

    # ---------------------------------------------------- numerical smoke tests (jax == numpy)
    def test_jax_matches_numpy_smoke(self):
        # A few complex ops: jax must agree with numpy (guards subtle backend divergence).
        base_np, var_np = bvf.t3_orthogonal_representations(self.x_np)
        v_np = t3m.T3Tangent(base_np, var_np)
        ww_np = tuple(np.asarray(w) for w in self.ww)

        def close(a, b):
            a, b = np.asarray(a), np.asarray(b)
            self.assertLessEqual(norm(a - b), tol * max(1.0, norm(b)))

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
