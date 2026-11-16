"""Target decision 2.9: the product of two full jets, carried out as a SCAN over the coefficients of one operand.

Decided on 2026-11-15: a term over the orders is the truncated product; one operand carries the order axis T
through the contraction and the other is taken one coefficient at a time; and that second part is a scan, not an
unrolled loop, so that the size of the compiled program does not grow with the order.

A scan body must have the same shapes at every iteration, so the triangular truncation cannot be done by slicing
(`a[:P - r]` has a different shape for each r).  Here it is done by a shift and a mask:

    c[t] = sum_r a[t - r] * b[r]      ->      for each r:  contract ALL of a with b[r], shift the result by r
                                              along the order axis, and zero the r slots that wrapped round.

This script checks, for the term 'Wa,aib,Wi->Wb' (mu_term) with two full jets:
  1. the scan gives the truncated product of jet_steps_check.py, to rounding;
  2. the size of the traced program is the same at every order, where the unrolled loop grows;
  3. it runs under jit.
It also counts the scalar multiplications each form does, since the mask throws part of the scan's work away.

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


def program_size(f, *args):                     # lines of the traced program, the scan body included
    return len(str(jax.make_jaxpr(f)(*args)).splitlines())


def main():
    rng = np.random.default_rng(0)
    W, ra, n, rb = 5, 3, 4, 2
    G = rng.standard_normal((ra, n, rb))
    print('%-6s %-22s %-22s %-18s %-18s %s' % ('order', 'scan against twin', 'loop against twin', 'program: scan',
                                               'program: loop', 'multiplications: scan / loop'))
    for order in (1, 2, 4, 8, 16):
        P = order + 1
        a, b = rng.standard_normal((P, W, ra)), rng.standard_normal((P, W, n))
        twin = J.mu_term(a, G, b)                                       # numpy, explicit sum over the orders
        by_scan = np.asarray(jax.jit(product_by_scan)(a, G, b))
        by_loop = np.asarray(jax.jit(product_by_loop)(a, G, b))
        rel = lambda x: np.linalg.norm(x - twin) / np.linalg.norm(twin)
        print('%-6d %-22s %-22s %-18d %-18d %d / %d = %.2f'
              % (order, 'ok' if rel(by_scan) < 1e-13 else 'DIFFERS %.0e' % rel(by_scan),
                 'ok' if rel(by_loop) < 1e-13 else 'DIFFERS %.0e' % rel(by_loop),
                 program_size(product_by_scan, a, G, b), program_size(product_by_loop, a, G, b),
                 P * P, P * (P + 1) // 2, (P * P) / (P * (P + 1) / 2)))
    print('("ok" = relative difference below 1e-13.  A multiplication here is one contraction for one pair of orders.)')


if __name__ == '__main__':
    main()
