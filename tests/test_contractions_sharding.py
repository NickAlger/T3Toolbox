# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The W-sharding invariant for contractions.py, pinned by compiling under multiple devices.

W is the SAMPLE axis, so it is the axis a user shards for data-parallel multi-GPU. The rule: within any
flattened group, only the LEFTMOST member can be sharded for free -- so W may absorb K and C (to its
right) but must never be folded with t or d (to its left). K-sharding is NOT supported today (the W+K
folds put K minor); that limitation is itself pinned below, so the docstring's claim is checked.
See the SHARDING block in contractions.py (which states the general rule and the three
levels it bites at) and
docs/contributor/batching_internals.md.

**Why this file exists.** The invariant is invisible to every other test: nothing else in the library
shards, and folding t into W is NUMERICALLY EXACT -- it is a pure reinterpretation, bit-identical to
the explicit form. So the numerical suite cannot see it, and it drifted (five lifts folded t, measured
3 all-gathers each). The only thing that can see it is the compiler: shard W, compile, and count the
collectives XLA had to insert. 0 = the reshape was free.

Runs on 4 VIRTUAL CPU devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`), which must be set
before jax initializes -- hence the module-level env manipulation and the import guard below.
"""
import os
import unittest

import numpy as np

import t3toolbox.backend.common as common

# Must precede the first jax device query. If jax was already initialized by another test module in the
# same process, the flag is ignored -- hence the device-count skip in setUpClass rather than an assert.
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                           ' --xla_force_host_platform_device_count=4').strip()


@unittest.skipUnless(common.jax_available, 'jax not available')
class TestWShardingInvariant(unittest.TestCase):
    """Every W-sharded contraction must compile to ZERO all-gathers."""

    @classmethod
    def setUpClass(cls):
        import jax
        if jax.device_count() < 2:
            raise unittest.SkipTest(
                'need >= 2 devices; XLA_FLAGS was ignored (jax already initialized in this process)')
        from jax.sharding import Mesh
        cls.jax = jax
        cls.mesh = Mesh(np.array(jax.devices()), axis_names=('m',))
        cls.rng = np.random.default_rng(0)

    def _all_gathers(self, fn, specs, *arrs):
        """Compile `fn` with the given PartitionSpecs and count all-gathers in the HLO."""
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        xs = [self.jax.device_put(jnp.asarray(a), NamedSharding(self.mesh, s))
              for a, s in zip(arrs, specs)]
        return self.jax.jit(fn).lower(*xs).compile().as_text().count('all-gather')

    def _check(self, label, fn, specs, arrs):
        with self.subTest(op=label):
            n = self._all_gathers(fn, specs, *arrs)
            self.assertEqual(
                n, 0,
                '%s inserted %d all-gather(s) with W sharded: something folded W together with an axis '
                'to its LEFT (t or d), so W is no longer the major member of a flattened group. See '
                'the SHARDING block in contractions.py.' % (label, n))

    def test_order_threaded_lifts_do_not_reshard(self):
        """The five sites that folded t into W (3 all-gathers each before the fix), plus the explicit
        t-carrying lifts they now delegate to."""
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        d, t, W, K, C, i, o = 2, 3, 8, 2, 2, 4, 5

        for label, fn, specs, arrs in [
            ('tWCi_KCio_to_tWKCo', lambda a, b: ctr.tWCi_KCio_to_tWKCo(a, b, 1),
             (P(None, 'm', None, None), P()),
             (r.standard_normal((t, W, C, i)), r.standard_normal((K, C, i, o)))),
            ('tWCi_Cio_to_tWCo', ctr.tWCi_Cio_to_tWCo,
             (P(None, 'm', None, None), P()),
             (r.standard_normal((t, W, C, i)), r.standard_normal((C, i, o)))),
            ('tWCo_Cio_to_tWCi', ctr.tWCo_Cio_to_tWCi,
             (P(None, 'm', None, None), P()),
             (r.standard_normal((t, W, C, o)), r.standard_normal((C, i, o)))),
            ('dtWCi_dCio_to_dtWCo', ctr.dtWCi_dCio_to_dtWCo,
             (P(None, None, 'm', None, None), P()),
             (r.standard_normal((d, t, W, C, i)), r.standard_normal((d, C, i, o)))),
            ('dtWKCi_dCio_to_dtWKCo', ctr.dtWKCi_dCio_to_dtWKCo,
             (P(None, None, 'm', None, None, None), P()),
             (r.standard_normal((d, t, W, K, C, i)), r.standard_normal((d, C, i, o)))),
            ('dtWCi_dKCio_to_dtWKCo', lambda a, b: ctr.dtWCi_dKCio_to_dtWKCo(a, b, 1),
             (P(None, None, 'm', None, None), P()),
             (r.standard_normal((d, t, W, C, i)), r.standard_normal((d, K, C, i, o)))),
            ('dtWCo_dCio_to_dtWCi', ctr.dtWCo_dCio_to_dtWCi,
             (P(None, None, 'm', None, None), P()),
             (r.standard_normal((d, t, W, C, o)), r.standard_normal((d, C, i, o)))),
            ('dtWKCo_dCio_to_dtWKCi', ctr.dtWKCo_dCio_to_dtWKCi,
             (P(None, None, 'm', None, None, None), P()),
             (r.standard_normal((d, t, W, K, C, o)), r.standard_normal((d, C, i, o)))),
        ]:
            self._check(label, fn, specs, arrs)

    def test_W_plus_K_folds_are_safe(self):
        """The folds that are CORRECT and must not be "fixed": they fuse W+K, and K is to W's right, so
        W stays major. Pinning them stops a future reader from splitting them for symmetry."""
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        d, t, W, K, C, i, o = 2, 3, 8, 2, 2, 4, 5

        for label, fn, specs, arrs in [
            ('WKCi_Cio_to_WKCo', ctr.WKCi_Cio_to_WKCo,
             (P('m', None, None, None), P()),
             (r.standard_normal((W, K, C, i)), r.standard_normal((C, i, o)))),
            ('tWKCi_Cio_to_tWKCo', ctr.tWKCi_Cio_to_tWKCo,
             (P(None, 'm', None, None, None), P()),
             (r.standard_normal((t, W, K, C, i)), r.standard_normal((C, i, o)))),
            ('tWKCo_Cio_to_tWKCi', ctr.tWKCo_Cio_to_tWKCi,
             (P(None, 'm', None, None, None), P()),
             (r.standard_normal((t, W, K, C, o)), r.standard_normal((C, i, o)))),
            ('dWKCi_dCio_to_dWKCo', ctr.dWKCi_dCio_to_dWKCo,
             (P(None, 'm', None, None, None), P()),
             (r.standard_normal((d, W, K, C, i)), r.standard_normal((d, C, i, o)))),
        ]:
            self._check(label, fn, specs, arrs)

    def test_K_sharding_is_not_supported_and_this_is_why(self):
        """Characterization, not aspiration. The W+K folds are free for W and NOT for K -- K is the
        MINOR member of that flatten. This pins the limitation the module docstring claims (level 2), so
        the claim is checked rather than asserted; if someone later un-fuses W+K for K-sharding, this
        test fails and points them at the docstring to update.

        The two sites that DON'T fuse (K is on the other operand, so it keeps its own letter) are free
        for K today -- which is also the recipe if K-sharding is ever wanted."""
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        d, t, W, K, C, i, o = 2, 3, 8, 4, 4, 4, 5

        # W+K fused -> K is minor -> costs collectives.
        x = r.standard_normal((d, t, W, K, C, i))
        y = r.standard_normal((d, C, i, o))
        self.assertEqual(0, self._all_gathers(ctr.dtWKCi_dCio_to_dtWKCo,
                                              (P(None, None, 'm', None, None, None), P()), x, y),
                         'W is major in the W+K fold and must stay free')
        self.assertGreater(self._all_gathers(ctr.dtWKCi_dCio_to_dtWKCo,
                                             (P(None, None, None, 'm', None, None), P()), x, y), 0,
                           'K-sharding a W+K fold is expected to reshard (K is minor). If this now '
                           'passes, W+K was un-fused: update the SHARDING block in contractions.py, '
                           'which documents K-sharding as unsupported.')

        # K on the other operand -> its own einsum letter -> already free.
        x2 = r.standard_normal((d, t, W, C, i))
        y2 = r.standard_normal((d, K, C, i, o))
        self.assertEqual(0, self._all_gathers(lambda a, b: ctr.dtWCi_dKCio_to_dtWKCo(a, b, 1),
                                              (P(), P(None, 'm', None, None, None)), x2, y2))

    def test_the_guard_can_fail(self):
        """The test above only means something if this style of check CAN fail. Fold t into W by hand --
        exactly what the five sites used to do -- and confirm XLA is forced to reshard."""
        from jax.sharding import PartitionSpec as P
        import jax.numpy as jnp
        r = self.rng
        d, t, W, C, i, o = 2, 3, 8, 2, 4, 5

        def folds_t_into_W(dtWCi, dCio):     # the pre-fix shape: reshape (t, W) -> one axis, t MAJOR
            dd, tt, ww, cc, ii = dtWCi.shape
            merged = dtWCi.reshape(dd, tt * ww, cc, ii)
            out = jnp.einsum('dWCi,dCio->dWCo', merged, dCio)
            return out.reshape(dd, tt, ww, cc, -1)

        n = self._all_gathers(folds_t_into_W, (P(None, None, 'm', None, None), P()),
                              r.standard_normal((d, t, W, C, i)), r.standard_normal((d, C, i, o)))
        self.assertGreater(n, 0, 'the t-into-W fold compiled with no collective -- this test is blind, '
                                 'and the invariant it claims to pin is unpinned')


if __name__ == '__main__':
    unittest.main()
