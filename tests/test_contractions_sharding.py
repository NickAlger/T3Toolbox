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
              'j': 2, 'c': 3}

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


if __name__ == '__main__':
    unittest.main()
