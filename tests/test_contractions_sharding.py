# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The sharding invariants for contractions.py, pinned by compiling under multiple devices.

The rule: **flatten only what einsum forces you to.** A block that is SHARED across operands needs an
einsum letter, so it must flatten its own axes (and then only its LEADING axis is shardable -- the one
remaining limit, characterized below). A block that is PASSIVE (on one operand, riding to the output)
needs no letter and rides as ``'...'``, flattening nothing -- so every one of its axes is shardable.
Nothing is fused with anything. W, K and C are ALL shardable today. See the SHARDING block in
contractions.py and docs/contributor/batching_internals.md.

**Why this file exists.** The invariant is invisible to every other test: nothing else in the library
shards, and a fusion is NUMERICALLY EXACT -- a pure reinterpretation, bit-identical to the unfused
form. So the numerical suite cannot see it, and it drifted repeatedly: five lifts folded t into W (3
all-gathers each); then six delegations fused W+K and eight merged K+C, invisible to the delegation
grep that caught the first five. Each inventory of "all the sites" has so far missed some -- hence a
check of the RULE (no named block fused with another) rather than a list. The only thing that can see
a fusion is the compiler: shard an axis, compile, count the collectives. 0 = the reshape was free.

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
    """Every sharded contraction must compile to ZERO all-gathers (W, K and C alike)."""

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

    def _check(self, label, fn, specs, arrs, block='W'):
        with self.subTest(op=label, sharded=block):
            n = self._all_gathers(fn, specs, *arrs)
            self.assertEqual(
                n, 0,
                '%s inserted %d all-gather(s) with %s sharded: something FUSED %s together with another '
                'block, so %s is no longer the major member of a flattened group. A passive block (one '
                'operand, riding to the output) must ride as \'...\' and flatten nothing. See the '
                'SHARDING block in contractions.py.' % (label, n, block, block, block))

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

    def test_W_sharding_on_the_ex_delegation_sites(self):
        """The six sites that used to fuse W+K by delegating to a WC-named twin. W was always free here
        (it was the major member of the W+K flatten); it must stay free now that nothing is fused."""
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
            ('dtWKCi_dCio_to_dtWKCo', ctr.dtWKCi_dCio_to_dtWKCo,
             (P(None, None, 'm', None, None, None), P()),
             (r.standard_normal((d, t, W, K, C, i)), r.standard_normal((d, C, i, o)))),
            ('dtWKCo_dCio_to_dtWKCi', ctr.dtWKCo_dCio_to_dtWKCi,
             (P(None, None, 'm', None, None, None), P()),
             (r.standard_normal((d, t, W, K, C, o)), r.standard_normal((d, C, i, o)))),
        ]:
            self._check(label, fn, specs, arrs, block='W')

    def test_K_sharding_is_free_on_the_ex_delegation_sites(self):
        """K-sharding IS supported. These six once fused W+K (delegating to a WC-named twin), which put K
        MINOR in the flatten and cost 3 all-gathers each -- this test used to assert that cost, as a
        characterization built to fail exactly when someone un-fused. Someone did: W and K are passive
        (they live on one operand and ride to the output), so they now ride as '...' with every axis
        intact, and K is free."""
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        d, t, W, K, C, i, o = 2, 3, 8, 4, 2, 4, 5   # K divisible by the 4 devices

        for label, fn, specs, arrs in [
            ('WKCi_Cio_to_WKCo', ctr.WKCi_Cio_to_WKCo,
             (P(None, 'm', None, None), P()),
             (r.standard_normal((W, K, C, i)), r.standard_normal((C, i, o)))),
            ('tWKCi_Cio_to_tWKCo', ctr.tWKCi_Cio_to_tWKCo,
             (P(None, None, 'm', None, None), P()),
             (r.standard_normal((t, W, K, C, i)), r.standard_normal((C, i, o)))),
            ('tWKCo_Cio_to_tWKCi', ctr.tWKCo_Cio_to_tWKCi,
             (P(None, None, 'm', None, None), P()),
             (r.standard_normal((t, W, K, C, o)), r.standard_normal((C, i, o)))),
            ('dWKCi_dCio_to_dWKCo', ctr.dWKCi_dCio_to_dWKCo,
             (P(None, None, 'm', None, None), P()),
             (r.standard_normal((d, W, K, C, i)), r.standard_normal((d, C, i, o)))),
            ('dtWKCi_dCio_to_dtWKCo', ctr.dtWKCi_dCio_to_dtWKCo,
             (P(None, None, None, 'm', None, None), P()),
             (r.standard_normal((d, t, W, K, C, i)), r.standard_normal((d, C, i, o)))),
            ('dtWKCo_dCio_to_dtWKCi', ctr.dtWKCo_dCio_to_dtWKCi,
             (P(None, None, None, 'm', None, None), P()),
             (r.standard_normal((d, t, W, K, C, o)), r.standard_normal((d, C, i, o)))),
        ]:
            self._check(label, fn, specs, arrs, block='K')

        # K on the OTHER operand -> it needs a real letter (it is shared-ish, not passive) -> also free.
        x2 = r.standard_normal((d, t, W, C, i))
        y2 = r.standard_normal((d, K, C, i, o))
        self.assertEqual(0, self._all_gathers(lambda a, b: ctr.dtWCi_dKCio_to_dtWKCo(a, b, 1),
                                              (P(), P(None, 'm', None, None, None)), x2, y2))

    def test_C_sharding_is_free_on_the_ex_internal_sites(self):
        """C-sharding IS supported. These four once merged K+C into a single einsum letter X
        ('Wo,WXa->WXao'), which put C MINOR in that flatten and cost 3 all-gathers. K and C are passive
        here (no operand carries C without K -- which is also why their split is not inferrable), so they
        now ride as '...' and C is free."""
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        d, W, K, C, a, o = 2, 8, 2, 4, 6, 5   # C divisible by the 4 devices

        Wo = r.standard_normal((W, o))
        WKCa = r.standard_normal((W, K, C, a))
        dWo = r.standard_normal((d, W, o))
        dWKCa = r.standard_normal((d, W, K, C, a))

        for label, fn, specs, arrs in [
            ('Wo_WKCa_to_WKCao', ctr.Wo_WKCa_to_WKCao,
             (P(), P(None, None, 'm', None)), (Wo, WKCa)),
            ('Wo_WKCa_to_KCao', ctr.Wo_WKCa_to_KCao,
             (P(), P(None, None, 'm', None)), (Wo, WKCa)),
            ('dWo_dWKCa_to_dWKCao', ctr.dWo_dWKCa_to_dWKCao,
             (P(), P(None, None, None, 'm', None)), (dWo, dWKCa)),
            ('dWo_dWKCa_to_dKCao', ctr.dWo_dWKCa_to_dKCao,
             (P(), P(None, None, None, 'm', None)), (dWo, dWKCa)),
        ]:
            self._check(label, fn, specs, arrs, block='C')

    def test_K_sharding_is_free_on_the_ex_internal_sites(self):
        """The same four, sharded on K (the other half of the old fused X = K+C block)."""
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        d, W, K, C, a, o = 2, 8, 4, 2, 6, 5   # K divisible by the 4 devices

        Wo = r.standard_normal((W, o))
        WKCa = r.standard_normal((W, K, C, a))
        dWo = r.standard_normal((d, W, o))
        dWKCa = r.standard_normal((d, W, K, C, a))

        for label, fn, specs, arrs in [
            ('Wo_WKCa_to_WKCao', ctr.Wo_WKCa_to_WKCao,
             (P(), P(None, 'm', None, None)), (Wo, WKCa)),
            ('Wo_WKCa_to_KCao', ctr.Wo_WKCa_to_KCao,
             (P(), P(None, 'm', None, None)), (Wo, WKCa)),
            ('dWo_dWKCa_to_dWKCao', ctr.dWo_dWKCa_to_dWKCao,
             (P(), P(None, None, 'm', None, None)), (dWo, dWKCa)),
            ('dWo_dWKCa_to_dKCao', ctr.dWo_dWKCa_to_dKCao,
             (P(), P(None, None, 'm', None, None)), (dWo, dWKCa)),
        ]:
            self._check(label, fn, specs, arrs, block='K')

    def test_K_and_C_sharding_on_the_assemble_dU_dxi_sites(self):
        """The four the unfusing plan's "complete, mechanically derived" inventory MISSED: the
        _assemble_dU_dxi[_d] helpers merged K+C into the letter X exactly like the four internal sites,
        but the inventory listed the _assemble_* helpers as already clean (true of _assemble_dU_eta and
        _assemble_dG_jet3, false of these two). Found by grepping for the fused letter rather than
        trusting the list -- the third time this file's inventories have missed a site."""
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        d, u, W, o, a = 2, 3, 8, 5, 6

        for block, K, C, s2, s4 in [
            ('K', 4, 2, P(None, None, 'm', None, None), P(None, None, None, 'm', None, None)),
            ('C', 2, 4, P(None, None, None, 'm', None), P(None, None, None, None, 'm', None)),
        ]:
            uWKCa = r.standard_normal((u, W, K, C, a))
            uWo = r.standard_normal((u, W, o))
            duWKCa = r.standard_normal((d, u, W, K, C, a))
            duWo = r.standard_normal((d, u, W, o))
            for label, fn, specs, arrs in [
                ('uWKCa_uWo_to_WKCao', ctr.uWKCa_uWo_to_WKCao, (s2, P()), (uWKCa, uWo)),
                ('uWKCa_uWo_to_KCao', ctr.uWKCa_uWo_to_KCao, (s2, P()), (uWKCa, uWo)),
                ('duWKCa_duWo_to_dWKCao', ctr.duWKCa_duWo_to_dWKCao, (s4, P()), (duWKCa, duWo)),
                ('duWKCa_duWo_to_dKCao', ctr.duWKCa_duWo_to_dKCao, (s4, P()), (duWKCa, duWo)),
            ]:
                self._check(label, fn, specs, arrs, block=block)

    def test_a_multi_axis_passive_block_is_shardable_on_ANY_axis(self):
        """The good one. A PASSIVE block rides as '...' and flattens NOTHING, so even its MINOR axis is
        free -- with W=(4,4) the old W+K fold cost 3 on W's minor axis, and giving W its own letter would
        STILL cost 3 (a letter forces the block to flatten its own axes). The plan accepted that as
        inherent; riding as '...' removes it. C is multi-axis here so a real reshape does happen -- of the
        C axes only.
        """
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        i, o = 4, 5
        x = r.standard_normal((4, 4, 4, 2, 2, i))    # W=(4,4), K=(4,), C=(2,2)
        y = r.standard_normal((2, 2, i, o))          # C=(2,2)

        for label, spec in [('W major', P('m', None, None, None, None, None)),
                            ('W minor', P(None, 'm', None, None, None, None)),
                            ('K', P(None, None, 'm', None, None, None))]:
            with self.subTest(sharded=label):
                n = self._all_gathers(ctr.WKCi_Cio_to_WKCo, (spec, P()), x, y)
                self.assertEqual(n, 0, 'WKCi_Cio_to_WKCo inserted %d all-gather(s) sharding %s: a '
                                       'passive block must flatten nothing.' % (n, label))

    def test_a_shared_block_still_only_shards_on_its_LEADING_axis(self):
        """Characterization of the ONE remaining limit, so it is checked rather than asserted. C is
        SHARED (it is on both operands, so einsum needs a letter for it), and a fixed subscript cannot
        spell a variable number of axes -- so a multi-axis C must flatten its own axes, and only its
        LEADING axis is free. This is inherent and ACCEPTED (the only escape is generating the subscript
        per call, which is rejected). If the minor case ever compiles to 0, this limit is gone: update
        the SHARDING block in contractions.py.
        """
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr
        r = self.rng
        i, o = 4, 5
        x = r.standard_normal((8, 4, 4, i))     # W=(8,), C=(4,4)
        y = r.standard_normal((4, 4, i, o))     # C=(4,4)

        self.assertEqual(0, self._all_gathers(ctr.WKCi_Cio_to_WKCo,
                                              (P(None, 'm', None, None), P('m', None, None, None)), x, y),
                         "C's LEADING axis is the major member of the C flatten and must be free")
        self.assertGreater(self._all_gathers(ctr.WKCi_Cio_to_WKCo,
                                             (P(None, None, 'm', None), P(None, 'm', None, None)), x, y), 0,
                           "C's MINOR axis is expected to reshard (C needs a letter, so it flattens its "
                           'own axes). If this now passes, the limit is gone: update the SHARDING block '
                           'in contractions.py.')

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


# ------------------------------------------------------------------------------------------------
# The shardability contract -- an automatic sweep over EVERY public contraction
# ------------------------------------------------------------------------------------------------
# The tests above pin the sites we know about. This pins the RULE, over every function in the
# module, because the sites we know about have never been the whole list: four hand-maintained
# inventories of this module have now been found wrong (the upstream survey, the docstring
# detector, the "complete, mechanically derived" unfusing inventory, and __all__ itself, which
# was missing 23 live functions -- including the family that hid the fusion bug). A uniform
# obligation has no list to be wrong about.

BLOCKS = ('W', 'K', 'C')          # the grouped indices the contract governs
_CORE = 'd'                       # the leading core/derivative axis
_AXIS_SIZE = {'d': 2, 't': 3, 'r': 3, 's': 3, 'u': 3, 'a': 2, 'i': 3, 'b': 5, 'o': 5, 'j': 2}


def _body_einsum(fn):
    """The einsum subscript literal in the body -- ground truth for an operand's letters."""
    import inspect
    import re
    m = re.search(r"'([a-zA-Z.]+(?:,[a-zA-Z.]+)*->[a-zA-Z.]*)'", inspect.getsource(fn))
    return m.group(1) if m else None


def _parse_operand(token, fn):
    """'WKCa' -> [('block','W'), ('block','K'), ('block','C'), ('axis','a')]; None if unparseable.

    `trs` is the one operand whose name is a family tag rather than a spec: its order-axis letters
    vary per function (and the output order's position inside it moves), so they come from the body
    einsum. It carries no grouped index, so the contract never shards it.
    """
    import re
    if token == 'trs':
        sub = _body_einsum(fn)
        if sub is None:
            return None
        return [('axis', ch) for ch in sub.split('->')[0].split(',')[0]]
    spec = []
    for i, ch in enumerate(token):
        if ch in BLOCKS:
            spec.append(('block', ch))
        elif ch == _CORE and i != 0:
            return None                      # 'd' is only ever the leading core axis
        elif re.fullmatch(r'[a-z]', ch):
            spec.append(('axis', ch))
        else:
            return None
    return spec or None


def _public_contractions():
    """Every public contraction, enumerated from the MODULE's own functions.

    Deliberately not `__all__`: that list was itself found wrong (78 listed, 101 defined), and the
    23 it omitted included `uWKCa_uWo_to_WKCao`, backed by one of the very helpers that hid the
    fusion bug from three enumerations. A sweep keyed on `__all__` would inherit the blind spot it
    exists to remove.
    """
    import inspect
    import t3toolbox.backend.contractions as ctr
    out = []
    for name, fn in inspect.getmembers(ctr, inspect.isfunction):
        if name.startswith('_') or fn.__module__ != ctr.__name__ or '_to_' not in name:
            continue
        out.append((name, fn))
    return out


@unittest.skipUnless(common.jax_available, 'jax not available')
class TestShardabilityContract(unittest.TestCase):
    """**Every grouped index must be shardable over its FIRST sub-axis.** Swept over every public
    contraction, with the block layout parsed from each function's name.

    This is EQUIVALENT to the no-fusing rule, not a proxy for it: fusing X with Y necessarily puts
    one of them to the right of the other, and the right-hand one's FIRST sub-axis is then non-major
    in the flatten, so it cannot be sharded for free. Every case we found confirms it -- t+W fused
    -> W's first axis pays; W+K -> K's first pays; K+C -> C's first pays.

    It also encodes exactly the limit that is accepted (`batching_internals.md`): it permits the
    within-block flatten a SHARED block requires -- only the leading sub-axis is claimed -- while
    forbidding cross-block fusion. And it checks a PROPERTY, not a form: a static name-vs-subscript
    check can be satisfied by writing the letters and flattening anyway; this cannot.

    Sizing matters and is not arbitrary. The block under test gets `(n_devices, 2)` so its leading
    axis is shardable and a real within-block flatten happens; every OTHER block gets size 2 rather
    than 1, because a size-1 neighbour still tiles correctly when fused and the check would pass
    vacuously.
    """

    @classmethod
    def setUpClass(cls):
        import jax
        if jax.device_count() < 2:
            raise unittest.SkipTest(
                'need >= 2 devices; XLA_FLAGS was ignored (jax already initialized in this process)')
        from jax.sharding import Mesh
        cls.jax = jax
        cls.n_dev = jax.device_count()
        cls.mesh = Mesh(np.array(jax.devices()), axis_names=('m',))
        cls.rng = np.random.default_rng(0)

    def _block_shape(self, block, target):
        return (self.n_dev, 2) if block == target else (2,)

    def _operand(self, spec, target):
        """(shape, PartitionSpec) for one operand, sharding `target`'s leading axis if present."""
        from jax.sharding import PartitionSpec as P
        shape, parts, hit = [], [], False
        for kind, name in spec:
            if kind == 'block':
                shp = self._block_shape(name, target)
                for j in range(len(shp)):
                    shape.append(shp[j])
                    parts.append('m' if (name == target and j == 0 and not hit) else None)
                    hit = hit or (name == target and j == 0)
            else:
                shape.append(_AXIS_SIZE.get(name, 3))
                parts.append(None)
        return tuple(shape), (P(*parts) if 'm' in parts else P())

    def test_every_grouped_index_shards_on_its_first_sub_axis(self):
        import inspect
        fns = _public_contractions()
        self.assertGreater(len(fns), 90, 'the sweep found suspiciously few contractions to check')

        checked = 0
        for name, fn in fns:
            sig = inspect.signature(fn)
            operands = [p for p in sig.parameters if p not in ('n_probe', 'n_frame')]
            specs = {t: _parse_operand(t, fn) for t in operands}
            if any(v is None for v in specs.values()):
                self.fail('%s: could not parse operand spec from the name -- the name IS the block '
                          'type signature here, so an unparseable one is a naming bug (or this '
                          'parser needs to learn a new form): %s'
                          % (name, [t for t, v in specs.items() if v is None]))

            present = {b for s in specs.values() for k, b in s if k == 'block'}
            for target in sorted(present):
                built = [self._operand(specs[t], target) for t in operands]
                arrs = [self.rng.standard_normal(shape) for shape, _ in built]
                pspecs = [ps for _, ps in built]

                ints = {}
                if 'n_probe' in sig.parameters:
                    ints['n_probe'] = len(self._block_shape('W', target))
                if 'n_frame' in sig.parameters:
                    ints['n_frame'] = len(self._block_shape('C', target))

                def call(*xs, _fn=fn, _names=tuple(operands), _ints=dict(ints)):
                    return _fn(**dict(zip(_names, xs)), **_ints)

                with self.subTest(op=name, sharded=target):
                    n = self._all_gathers(call, pspecs, *arrs)
                    self.assertEqual(
                        n, 0,
                        "%s inserted %d all-gather(s) with %s's LEADING sub-axis sharded. The "
                        'contract is that every grouped index shards over its first sub-axis; a '
                        'failure here means %s was FUSED with a block to its left, so its first '
                        'axis is no longer major in the flattened group. Fix the contraction (let '
                        "the passive block ride as '...'), do not weaken this test." % (
                            name, n, target, target))
                    checked += 1

        # the sweep is only worth anything if it actually swept
        self.assertGreater(checked, 150,
                           'only %d (function, block) pairs were checked -- the sweep is not '
                           'covering the module' % checked)

    def _all_gathers(self, fn, specs, *arrs):
        """Compile `fn` with the given PartitionSpecs and count all-gathers in the HLO."""
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        xs = [self.jax.device_put(jnp.asarray(a), NamedSharding(self.mesh, s))
              for a, s in zip(arrs, specs)]
        return self.jax.jit(fn).lower(*xs).compile().as_text().count('all-gather')


if __name__ == '__main__':
    unittest.main()
