"""Target decision 2.9: the product of two full jets, carried out as a SCAN over the coefficients of one operand.

Decided on 2026-11-15: a term over the orders is the truncated product; one operand carries the order axis T
through the contraction and the other is taken one coefficient at a time; and that second part is a scan, not an
unrolled loop, so that the size of the compiled program does not grow with the order.

A scan body must have the same shapes at every iteration, so the triangular truncation cannot be done by slicing
(`a[:P - r]` has a different shape for each r).  Here it is done by a shift and a mask:

    c[t] = sum_r a[t - r] * b[r]      ->      for each r:  contract ALL of a with b[r], shift the result by r
                                              along the order axis, and zero the r slots that wrapped round.

The mask throws away the pairs of orders that wrapped round: about half of them.  A third form avoids that: the
pairs of orders the product needs are listed in a static table, and the scan takes them P at a time.

This script compares the three forms for the term 'Wa,aib,Wi->Wb' (mu_term) with two full jets: the pairs of
orders each contracts, the size of its traced program, and whether it gives the truncated product of
jet_steps_check.py under jit.

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


def product_by_scan(a, G, b):
    """c[t] = sum_r contract(a[t-r], G, b[r]).  a: (P, W, ra) carries T; b: (P, W, n) is scanned."""
    P = a.shape[0]
    orders = jnp.arange(P)

    def body(c, r_and_b):
        r, b_r = r_and_b
        full = jnp.einsum('tWa,aib,Wi->tWb', a, G, b_r)                # slot i holds a[i] * b[r]: it belongs at t = i + r
        shifted = jnp.roll(full, r, axis=0)
        keep = (orders >= r).reshape((P, 1, 1))
        return c + jnp.where(keep, shifted, 0.0), None

    c0 = jnp.zeros((P, a.shape[1], G.shape[2]), a.dtype)
    c, _ = jax.lax.scan(body, c0, (orders, b))
    return c


def product_by_loop(a, G, b):
    """The same product with the loop over r unrolled and the truncation done by slicing."""
    P = a.shape[0]
    c = jnp.zeros((P, a.shape[1], G.shape[2]), a.dtype)
    for r in range(P):
        c = c.at[r:].add(jnp.einsum('tWa,aib,Wi->tWb', a[:P - r], G, b[r]))
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
        vals = jnp.einsum('pWa,aib,pWi->pWb', a[i], G, b[r])            # p runs over the P pairs of this chunk
        return c.at[t].add(jnp.where(ok[:, None, None], vals, 0.0)), None

    c0 = jnp.zeros((P, a.shape[1], G.shape[2]), a.dtype)
    c, _ = jax.lax.scan(body, c0, (jnp.asarray(i_of_a), jnp.asarray(r_of_b), jnp.asarray(t_out), jnp.asarray(valid)))
    return c


def program_size(f, *args):                     # lines of the traced program, the scan body included
    return len(str(jax.make_jaxpr(f)(*args)).splitlines())


def main():
    rng = np.random.default_rng(0)
    W, ra, n, rb = 5, 3, 4, 2
    G = rng.standard_normal((ra, n, rb))
    forms = (('loop, slices', product_by_loop), ('scan, shift and mask', product_by_scan),
             ('scan over the pair table', product_by_pair_scan))
    print('Pairs of orders contracted, for the product of two full jets.  The truncated product needs the triangle.')
    print('%-6s %-10s | %-28s | %-28s | %-28s' % (('order', 'triangle') + tuple(name for name, _ in forms)))
    print('%-6s %-10s | %-28s | %-28s | %-28s' % (('', '') + ('pairs   program   correct',) * 3))
    for order in (1, 2, 3, 4, 8, 16):
        P = order + 1
        a, b = rng.standard_normal((P, W, ra)), rng.standard_normal((P, W, n))
        twin = J.mu_term(a, G, b)                                       # numpy, explicit sum over the orders
        triangle = P * (P + 1) // 2
        pairs = (triangle, P * P, triangle + pair_table(P)[5])
        cells = []
        for (name, f), n_pairs in zip(forms, pairs):
            out = np.asarray(jax.jit(f)(a, G, b))
            ok = np.linalg.norm(out - twin) / np.linalg.norm(twin) < 1e-13
            cells.append('%-7d %-9d %s' % (n_pairs, program_size(f, a, G, b), 'yes' if ok else 'NO'))
        print('%-6d %-10d | %-28s | %-28s | %-28s' % ((order, triangle) + tuple(cells)))
    print('(program = lines of the traced program, scan body included.  correct = relative difference from the explicit')
    print(' sum below 1e-13, under jit.)')


if __name__ == '__main__':
    main()
