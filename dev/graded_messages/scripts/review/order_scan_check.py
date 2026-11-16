"""Target decision 2.9: the product of two full jets over the orders, in three forms.

    c[t] = sum over i + r = t of contract(a[i], G, b[r])        P = order + 1 coefficients in each jet

  loop          for each r: contract G with b[r], then with the slice a[:P-r].  The program has P copies.
  carried shift a scan over r: contract G with b[r], then with ALL of a carried jet, which starts as `a` and is
                shifted one slot along the order axis after each step (zeros come in at the bottom).  This is the
                library's present scheme (backend/sampling_derivatives.py, `_eta_jets_order_step` and
                `_adj_src_order_step`) with the binomial weights gone, as they are for Taylor coefficients.
  pair table    a scan over a static table of the pairs (i, r) the product needs, P pairs at a time.

Two different contractions are involved, and they do not cost the same:
  E, the expensive one: the core with one coefficient of b.   W * ra * n * rb multiplications.
  c, the cheap one:     that product with one coefficient of a.   W * ra * rb multiplications: n times fewer.

  loop           P * E  +  P (P + 1) / 2 * c        the triangle of cheap contractions
  carried shift  P * E  +  P^2 * c                  the square of cheap ones; the extra ones meet zeros
  pair table     about P (P + 1) / 2 * (E + c)      the EXPENSIVE contraction is repeated for every pair

So the carried shift costs more than the loop only in the cheap contraction, by the factor
(P - 1) / 2 / (n + (P + 1) / 2) of the whole, and the pair table is the worst of the three.

The script checks that each form gives the explicit sum of jet_steps_check.py under jit, shows the size of each
traced program, and evaluates the counts above at two sets of sizes.

Needs jax.  Reads nothing from T3Toolbox.   Usage:  python3 -I order_scan_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jet_steps_check as J

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def product_by_carried_shift(a, G, b):
    """A scan over the coefficients of b; the jet `a` rides the carry and is shifted one slot per step."""
    def body(carry, b_r):
        c, a_shifted = carry
        Gb = jnp.einsum('aib,Wi->Wab', G, b_r)                           # E: once per coefficient of b
        c = c + jnp.einsum('tWa,Wab->tWb', a_shifted, Gb)                # c: against every slot of the carried jet
        a_shifted = jnp.concatenate([jnp.zeros_like(a_shifted[:1]), a_shifted[:-1]], axis=0)
        return (c, a_shifted), None

    c0 = jnp.zeros((a.shape[0], a.shape[1], G.shape[2]), a.dtype)
    (c, _), _ = jax.lax.scan(body, (c0, a), b)
    return c


def product_by_loop(a, G, b):
    """The loop over r unrolled, with the truncation done by slicing."""
    P = a.shape[0]
    c = jnp.zeros((P, a.shape[1], G.shape[2]), a.dtype)
    for r in range(P):
        Gb = jnp.einsum('aib,Wi->Wab', G, b[r])                          # E
        c = c.at[r:].add(jnp.einsum('tWa,Wab->tWb', a[:P - r], Gb))      # c: against the P - r slots that stay in range
    return c


def pair_table(P):
    """The pairs of orders (i of a, r of b) that a truncated product needs, in chunks of P, as static numpy tables."""
    pairs = [(i, r, i + r) for r in range(P) for i in range(P - r)]
    n_chunks = -(-len(pairs) // P)
    pad = n_chunks * P - len(pairs)
    table = np.array(pairs + [(0, 0, 0)] * pad).reshape((n_chunks, P, 3))
    valid = np.array([True] * len(pairs) + [False] * pad).reshape((n_chunks, P))
    return table[..., 0], table[..., 1], table[..., 2], valid, len(pairs), pad


def product_by_pair_scan(a, G, b):
    """The same product as a scan over the table of needed pairs, P pairs at a time: nothing is thrown away."""
    P = a.shape[0]
    i_of_a, r_of_b, t_out, valid, _, _ = pair_table(P)

    def body(c, chunk):
        i, r, t, ok = chunk
        Gb = jnp.einsum('aib,pWi->pWab', G, b[r])                        # E, once for EVERY pair of the chunk
        vals = jnp.einsum('pWa,pWab->pWb', a[i], Gb)                     # c
        return c.at[t].add(jnp.where(ok[:, None, None], vals, 0.0)), None

    c0 = jnp.zeros((P, a.shape[1], G.shape[2]), a.dtype)
    c, _ = jax.lax.scan(body, c0, (jnp.asarray(i_of_a), jnp.asarray(r_of_b), jnp.asarray(t_out), jnp.asarray(valid)))
    return c


def program_size(f, *args):                     # lines of the traced program, the scan body included
    return len(str(jax.make_jaxpr(f)(*args)).splitlines())


def counts(P, W, ra, n, rb):                  # multiplications of each form, by the formulas of the docstring
    E, c, triangle = W * ra * n * rb, W * ra * rb, P * (P + 1) // 2
    n_chunks = -(-triangle // P)
    return P * E + triangle * c, P * E + P * P * c, n_chunks * P * (E + c)


def main():
    rng = np.random.default_rng(0)
    W, ra, n, rb = 5, 3, 4, 2
    G = rng.standard_normal((ra, n, rb))
    forms = (('loop', product_by_loop), ('carried shift', product_by_carried_shift), ('pair table', product_by_pair_scan))
    print('1. Each form against the explicit sum, under jit, and the lines of its traced program (scan body included)')
    print('   %-6s %-22s %-22s %-22s' % (('order',) + tuple(name for name, _ in forms)))
    for order in (1, 2, 4, 8, 16):
        P = order + 1
        a, b = rng.standard_normal((P, W, ra)), rng.standard_normal((P, W, n))
        twin = J.mu_term(a, G, b)                                       # numpy, explicit sum over the orders
        cells = []
        for name, f in forms:
            out = np.asarray(jax.jit(f)(a, G, b))
            ok = np.linalg.norm(out - twin) / np.linalg.norm(twin) < 1e-13
            cells.append('%s, %d lines' % ('correct' if ok else 'WRONG', program_size(f, a, G, b)))
        print('   %-6d %-22s %-22s %-22s' % ((order,) + tuple(cells)))
    print('2. Multiplications, relative to the loop')
    for label, sizes in (('W = 1000, bonds 64, mode rank 32', (1000, 64, 32, 64)), ('W = 1000, bonds 4, mode rank 4', (1000, 4, 4, 4))):
        print('   %s' % label)
        print('   %-6s %-16s %-16s' % ('order', 'carried shift', 'pair table'))
        for order in (1, 2, 4, 8, 16):
            loop, shift, table = counts(order + 1, *sizes)
            print('   %-6d %-16s %-16s' % (order, '%.2f' % (shift / loop), '%.2f' % (table / loop)))


if __name__ == '__main__':
    main()
