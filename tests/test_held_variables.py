# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""``backend.sampling_variables.HeldVariables``: the record of held sampling variables.

It reads like a dictionary over its filled slots; an unknown name raises at construction, in ``with_values`` and at
lookup; ``with_values`` returns a new record and leaves the old one as it was; ``==`` is refused, as for every
array-carrying class of the library; where jax is present it is a pytree whose leaves are the filled slots, and the
engine takes it under ``jax.jit``. (Imported and used with no jax: ``test_jax_absent.py``.)
"""
import unittest

import numpy as np

import t3toolbox.backend.sampling_variables as sv

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

SLOTS = ('xi', 'mu', 'nu', 'nu_L', 'eta', 'eta_D', 'z', 'r')


def arrays(n, size=2):
    return tuple(np.full((3, size), float(k)) for k in range(n))


class TestHeldVariables(unittest.TestCase):

    def test_the_slots(self):
        self.assertEqual(sv.HeldVariables.slots(), SLOTS)
        empty = sv.HeldVariables()
        self.assertEqual((empty.keys(), len(empty), list(empty)), ((), 0, []))
        for slot in SLOTS:
            self.assertIsNone(getattr(empty, slot))

    def test_it_reads_like_a_dictionary_over_the_filled_slots(self):
        mu, r = arrays(4), arrays(3)
        held = sv.HeldVariables(r=r, mu=mu)
        self.assertEqual(held.keys(), ('mu', 'r'))                      # in the order of the slots
        self.assertEqual(list(held), ['mu', 'r'])
        self.assertEqual(len(held), 2)
        self.assertIs(held['mu'], mu)
        self.assertIs(held['r'], r)
        self.assertTrue('mu' in held and 'r' in held)
        self.assertFalse('nu' in held)                                  # a slot, not filled
        self.assertFalse('w' in held)                                   # not a slot: in is a membership test
        self.assertEqual(dict((name, held[name]) for name in held).keys(), {'mu', 'r'})

    def test_an_unknown_name_raises(self):
        with self.assertRaises(TypeError):
            sv.HeldVariables(w=arrays(3))                               # the sample is an argument, not a slot
        with self.assertRaises(TypeError):
            sv.HeldVariables(muu=arrays(4))
        held = sv.HeldVariables(mu=arrays(4))
        with self.assertRaisesRegex(KeyError, "'g' is not a variable that can be held"):
            held['g']
        with self.assertRaisesRegex(KeyError, "the slot 'nu' is not filled"):
            held['nu']
        with self.assertRaisesRegex(TypeError, "'g', 'w' are not a variable that can be held"):
            held.with_values(g=1, w=2)

    def test_with_values_returns_a_new_record(self):
        mu, nu = arrays(4), arrays(4, 3)
        held = sv.HeldVariables(mu=mu)
        more = held.with_values(nu=nu)
        self.assertIsNot(more, held)
        self.assertEqual((held.keys(), more.keys()), (('mu',), ('mu', 'nu')))
        self.assertIs(more['mu'], mu)
        fewer = more.with_values(mu=None)                               # None empties a slot
        self.assertEqual((fewer.keys(), more.keys()), (('nu',), ('mu', 'nu')))
        with self.assertRaises(Exception):
            held.mu = nu                                                # frozen

    def test_equality_is_refused(self):
        held = sv.HeldVariables(mu=arrays(4))
        with self.assertRaises(TypeError):
            held == sv.HeldVariables(mu=arrays(4))
        with self.assertRaises(TypeError):
            hash(held)

    @unittest.skipUnless(HAS_JAX, 'jax not available')
    def test_the_pytree_round_trip(self):
        mu, r = arrays(4), arrays(3)
        held = sv.HeldVariables(mu=mu, r=r)
        leaves, treedef = jax.tree_util.tree_flatten(held)
        self.assertEqual(len(leaves), 7)                                # the arrays of the filled slots
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertIsInstance(back, sv.HeldVariables)
        self.assertEqual(back.keys(), ('mu', 'r'))
        for a, b in zip(back['mu'] + back['r'], mu + r):
            self.assertIs(a, b)
        doubled = jax.tree_util.tree_map(lambda x: 2 * x, held)
        self.assertTrue(np.array_equal(doubled['mu'][3], 2 * mu[3]))
        self.assertEqual(jax.tree_util.tree_structure(sv.HeldVariables(mu=mu)),
                         jax.tree_util.tree_structure(sv.HeldVariables(mu=arrays(4))))
        self.assertNotEqual(jax.tree_util.tree_structure(sv.HeldVariables(mu=mu)), treedef)

    @unittest.skipUnless(HAS_JAX, 'jax not available')
    def test_the_engine_takes_it_under_jit(self):
        """A held ``mu`` passed into a jitted call is a pytree argument: its arrays are traced, and the result is
        the eager one."""
        rng = np.random.default_rng(0)
        N, nD, nU, rL, rR = (4, 5, 3), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
        frame = ([rng.standard_normal((nD[i], N[i])) for i in range(3)],
                 [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)],
                 [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)],
                 [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)])
        ww = [rng.standard_normal((2, n)) for n in N]
        (mu,) = sv.sampling_variables('probe', frame, ww, ('mu',))
        (eta,) = sv.sampling_variables('probe', frame, ww, ('eta',))
        jitted = jax.jit(lambda held: sv.sampling_variables('probe', frame, ww, ('eta',), held=held)[0])(
            sv.HeldVariables(mu=tuple(jnp.asarray(m) for m in mu)))
        for a, b in zip(jitted, eta):
            self.assertLessEqual(np.linalg.norm(np.asarray(a) - b), 1e-5 * np.linalg.norm(b))


if __name__ == '__main__':
    unittest.main()
