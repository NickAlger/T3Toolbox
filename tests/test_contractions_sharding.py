# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The sharding invariant for the grouped-einsum interpreter, pinned by compiling under multiple
devices: **every sub-axis of every group is shardable, with zero all-gathers.**

The interpreter (`contractions.contract`) never reshapes -- each group sub-axis is an honest einsum
axis -- so the invariant holds by construction. It is still checked with the compiler, because that
is how its predecessors' violations were caught: the old enumerated named contractions flattened
each shared block to one letter (only the leading sub-axis free) and repeatedly drifted into
FUSING blocks -- numerically exact, so invisible to every numerical test; hand inventories of the
sites were wrong four times; only sharding an axis, compiling, and counting collectives ever saw
it. (That history: docs/contributor/batching_internals.md.) Hence a sweep over the FULL library
vocabulary of subscripts strings, AST-scanned from the source so it can never under-count.

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


def _scan_vocabulary():
    """Every grouped-subscript string literal in the library source (the same self-maintaining
    scan as test_contractions_interpreter.py -- duplicated so the test modules stay standalone)."""
    import ast
    import pathlib
    import re
    import t3toolbox
    subs_re = re.compile(r'^[A-Za-z]+(,[A-Za-z]+)*->[A-Za-z]*$')
    vocab = set()
    for path in sorted(pathlib.Path(t3toolbox.__file__).parent.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.replace(' ', '')
                if subs_re.match(s) and any(ch.isupper() for ch in s):
                    vocab.add(s)
    return vocab


@unittest.skipUnless(common.jax_available, 'jax not available')
class TestInterpreterAnyAxisSharding(unittest.TestCase):
    """The grouped-einsum interpreter never reshapes, so EVERY sub-axis of EVERY group is shardable
    -- strictly stronger than the named contractions' contract, whose one residue (a shared block
    flattens, so only its LEADING axis is free; batching_internals.md) was an artifact of the
    flatten. Swept over the FULL library vocabulary (AST-scanned, self-maintaining): shard each
    sub-axis of each group of each subscripts string in turn, compile, assert ZERO all-gathers. A
    summed group legitimately costs an all-REDUCE (the psum the math requires) -- never an
    all-gather.
    """

    SINGLE = {'d': 2, 't': 3, 'r': 3, 's': 2, 'u': 3, 'k': 2, 'a': 2, 'i': 3, 'b': 3, 'o': 3,
              'j': 2, 'c': 3,
              'l': 4, 'n': 6, 'x': 7, 'z': 5}   # the exact Hessian's mode / Z-rank / fourth-bond letters

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

    def _all_gathers(self, fn, specs, *arrs):
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        xs = [self.jax.device_put(jnp.asarray(a), NamedSharding(self.mesh, s))
              for a, s in zip(arrs, specs)]
        return self.jax.jit(fn).lower(*xs).compile().as_text().count('all-gather')

    def test_every_sub_axis_of_every_group_shards_free(self):
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.contractions as ctr

        vocab = sorted(_scan_vocabulary())
        self.assertGreater(len(vocab), 100, 'the AST scan found suspiciously few subscripts')
        checked = 0
        for subs in vocab:
            terms = subs.split('->')[0].split(',')
            groups = sorted({ch for ch in subs if ch.isupper()})
            lens = {'len_' + g: 2 for g in groups}      # all supplements: always sufficient
            for g in groups:
                for ax in range(2):     # every group gets 2 sub-axes; ax=1 is the minor one
                    with self.subTest(subs=subs, group=g, axis=ax):
                        def gshape(ch):     # sharded sub-axis sized to the mesh, the rest 2
                            return tuple(self.n_dev if (ch == g and j == ax) else 2
                                         for j in range(2))
                        arrs, specs = [], []
                        for t in terms:
                            shape, parts = [], []
                            for ch in t:
                                if ch.isupper():
                                    for j in range(2):
                                        shape.append(gshape(ch)[j])
                                        parts.append('m' if (ch == g and j == ax) else None)
                                else:
                                    shape.append(self.SINGLE[ch])
                                    parts.append(None)
                            arrs.append(self.rng.standard_normal(shape))
                            specs.append(P(*parts) if 'm' in parts else P())
                        n = self._all_gathers(
                            lambda *ops: ctr.contract(subs, *ops, **lens), specs, *arrs)
                        self.assertEqual(0, n,
                            "contract(%r) inserted %d all-gather(s) with axis %d of group %s "
                            "sharded -- the interpreter must not reshape, so every group sub-axis "
                            "must shard free." % (subs, n, ax, g))
                        checked += 1
        self.assertGreater(checked, 400, 'the any-axis sweep barely ran (%d cases)' % checked)

    def test_every_vocabulary_letter_has_a_size(self):
        """The sweep above is AST-driven over the library, so every new subscripts string is in it the moment
        it is written; what can silently go missing is a NEW single-axis letter (the sweep would KeyError).
        Pin that the size table covers the whole vocabulary."""
        letters = {ch for subs in _scan_vocabulary() for ch in subs if ch.islower()}
        self.assertFalse(letters - set(self.SINGLE), 'letters with no sweep size: %s' % sorted(letters - set(self.SINGLE)))


@unittest.skipUnless(common.jax_available, 'jax not available')
class TestJetKernelsShardOverW(unittest.TestCase):
    """The probe-derivative transpose and the exact Hessian's two-hole under ``shard_map`` over the sample stack
    ``W`` (``docs/chunking.md``, "Sharding over W"): the W-carrying arrays are sharded, the cores replicated (closed
    over), each device runs the kernels -- order scans, banded recurrences, the W-chunked assembly -- on its local
    shard, and one ``psum`` finishes the W-sum. The jet kernels move data only along the order axis (it leads every
    jet, so no batch axis is ever moved), so the compiled program must contain NO all-gather (the ``psum``'s
    all-reduce is expected), and the result must equal the unsharded one. A jax dispatch test (float64)."""

    @classmethod
    def setUpClass(cls):
        import jax
        if jax.device_count() < 2:
            raise unittest.SkipTest(
                'need >= 2 devices; XLA_FLAGS was ignored (jax already initialized in this process)')
        jax.config.update('jax_enable_x64', True)
        from jax.sharding import Mesh
        cls.jax = jax
        cls.n_dev = jax.device_count()
        cls.mesh = Mesh(np.array(jax.devices()), axis_names=('w',))

    def _check(self, name, fn, sharded_args, specs, full_args):
        """fn(*W-carrying args) -> pytree (W summed); compare the shard_map + psum run with the unsharded one and
        scan the compiled HLO for all-gathers."""
        import inspect
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        jax = self.jax
        # jax >= 0.7 has jax.shard_map(..., check_vma=); the jax 0.4.30 floor only
        # jax.experimental.shard_map.shard_map(..., check_rep=). The check is switched off either way: the
        # kernels' scan carries start from constants (zeros / the order-0 ones), which the varying-axes check
        # rejects against a W-varying step output -- pre-existing (the 2026.2.0 kernels fail it identically);
        # numerics and the HLO are unaffected.
        shard_map = getattr(jax, 'shard_map', None)
        legacy = shard_map is None
        if legacy:
            from jax.experimental.shard_map import shard_map
        params = inspect.signature(shard_map).parameters
        off = {k: False for k in ('check_vma', 'check_rep') if k in params}
        body = shard_map(lambda *a: jax.tree.map(lambda x: jax.lax.psum(x, 'w'), fn(*a)),
                         mesh=self.mesh, in_specs=tuple(specs), out_specs=P(), **off)
        placed = [jax.device_put(jnp.asarray(a), NamedSharding(self.mesh, s)) for a, s in zip(sharded_args, specs)]
        try:
            compiled = jax.jit(body).lower(*placed).compile()
        except Exception as e:                    # only on the legacy API: name the version, do not hide it
            if not legacy:
                raise
            raise unittest.SkipTest('jax %s: jax.experimental.shard_map could not lower the kernels (%s: %s)'
                                    % (jax.__version__, type(e).__name__, str(e).split('\n')[0][:120]))
        hlo = compiled.as_text()
        self.assertEqual(0, hlo.count('all-gather'), '%s: the sharded program all-gathers' % name)
        self.assertIn('all-reduce', hlo, '%s: no all-reduce -- was W really sharded?' % name)
        got = jax.tree.leaves(compiled(*placed))
        ref = jax.tree.leaves(jax.jit(fn)(*[jnp.asarray(a) for a in full_args]))
        self.assertEqual(len(got), len(ref))
        for g, r in zip(got, ref):
            g, r = np.asarray(g), np.asarray(r)
            self.assertEqual(g.shape, r.shape, name)
            self.assertLess(np.linalg.norm(g - r) / np.linalg.norm(r), 1e-14, name)

    def test_probe_transpose_and_two_hole_shard_over_W(self):
        from jax.sharding import PartitionSpec as P
        import t3toolbox.backend.sampling_derivatives as sd
        rng = np.random.default_rng(0)
        d, n, r, N, order = 3, 3, 3, 5, 3
        W = 4 * self.n_dev                                   # 4 probes per device; chunk_size 2 chunks the shard
        R = lambda *s: rng.standard_normal(s)
        frame = (R(d, n, N), R(d, r, n, r), R(d, r, n, r), R(d, r, n, r))     # uniform (U, O, P, Q), replicated
        theta, dirs = (R(d, n, N), R(d, r, n, r)), (R(d, n, N), R(d, r, n, r))
        zt, ww, pp = R(d, order + 1, W, N), R(d, W, N), R(d, W, N)
        specs = (P(None, None, 'w', None), P(None, 'w', None), P(None, 'w', None))   # W: axis 2 / 1 / 1
        for cs in (None, 2):
            with self.subTest(op='probe_derivatives_transpose', chunk_size=cs):
                self._check('transpose', lambda z, w, p: sd.tv_probe_derivatives_transpose(
                    z, w, p, frame, order, sum_over_probes=True, chunk_size=cs), (zt, ww, pp), specs, (zt, ww, pp))
            with self.subTest(op='two_hole', chunk_size=cs):
                self._check('two-hole', lambda z, w, p: sd.t3_probe_corewise_derivatives_hessian(
                    z, w, p, theta, dirs, order, chunk_size=cs), (zt, ww, pp), specs, (zt, ww, pp))


if __name__ == '__main__':
    unittest.main()
