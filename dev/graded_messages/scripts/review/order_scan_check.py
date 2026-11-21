"""Target decision 2.9: the product of two full jets over the orders.

    c[t] = sum over i + r = t of contract(a[i], G, b[r])        P = order + 1 coefficients in each jet

DECIDED (the maintainer, 2026-11-15, 19:32): the target has TWO forms, chosen by the user through the plan.

  'scan' (the default)  A scan over the coefficients of b: contract G with b[r], then with ALL of a carried
                        jet, which starts as `a` and is shifted one slot along the order axis after each step
                        (zeros come in at the bottom).  This is the library's present scheme
                        (backend/sampling_derivatives.py, `_eta_jets_order_step`, `_adj_src_order_step`) with
                        the binomial weights gone, as they are for Taylor coefficients.
  'loop'                A Python loop over r: contract G with b[r], then with the slice a[:P-r].

HOW THE SCAN IS WRITTEN (the maintainer, 2026-11-15, 19:50): through the library's `xscan`, obtained from
`get_backend(is_uniform, use_jax)`.  It is a real `jax.lax.scan` when jax is used AND the layout is uniform, and
a Python loop on numpy or on the ragged layout.  numpy or jax is inferred from the inputs (`tree_contains_jax`),
at the lowest level.  The layout is deduced from the input types by the caller and handed down, because one
core's arrays look the same in both layouts (the library does the same: `_eta_jets_step_uniform` and
`_eta_jets_step_ragged` both call `_eta_jets_core(data, is_uniform)`).  The body is a module-level function that
closes over nothing: the core rides the carry.  The contractions go through the grouped interpreter `contract`, so the `W` and `C` stacks ride along.

Two different contractions are involved, and they do not cost the same:
  E, the expensive one: the core with one coefficient of b.   W * ra * n * rb multiplications.
  c, the cheap one:     that product with one coefficient of a.   W * ra * rb multiplications: n times fewer.

  'loop'         P * E  +  P (P + 1) / 2 * c        the triangle of cheap contractions
  'scan'         P * E  +  P^2 * c                  the square of cheap ones; the extra ones meet zeros
  pair table     about P (P + 1) / 2 * (E + c)      the EXPENSIVE contraction is repeated for every pair

The pair table (a scan over a static table of the needed pairs) was a third form Claude proposed and withdrew.
It is kept here, in jax only, as the record of the comparison.  It is not in the target.

The script checks
  1. each form against the explicit sum of jet_steps_check.py under jit, and the size of each traced program;
     'scan' is shown on both layouts, since only the uniform one is a real scan;
  2. the counts above at two sets of sizes;
  3. the option on all four backends (numpy or jax, ragged or uniform): the two settings agree to rounding, the
     output is of the kind of the inputs (numpy in, numpy out; jax in, jax out), and float32 stays float32.

Needs jax.   Usage:  python3 -I order_scan_check.py /path/to/T3Toolbox
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jet_steps_check as J

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


# The library's dispatch functions.  Bound by `load_library` (the checkout's path is an argument of the script);
# in the library these are ordinary module-level imports, as in backend/probing.py.
get_backend = tree_contains_jax = contract = None


def load_library(repo):
    global get_backend, tree_contains_jax, contract
    sys.path.insert(0, repo)
    from t3toolbox.backend import common
    from t3toolbox.backend import contractions
    get_backend, tree_contains_jax, contract = common.get_backend, common.tree_contains_jax, contractions.contract
    return common


# ---- the two forms of the target, written as the library writes its kernels --------------------------------
# What is as in the library: every body is a module-level function that closes over nothing (the core rides the
# carry; `xnp` is re-derived from the body's own arguments); numpy or jax comes from `tree_contains_jax`; the scan
# comes from `get_backend`; the stacks are groups of `contract`.   docs/contributor/scan_body_principles.md
# What is prototype only: the names `a`, `G`, `b`; one `W` axis, no `K`, no `C` in the test; and the setting as
# an argument of `product_of_full_jets`.  In the library a static that no operand carries reaches a body through
# step variants or a memoized factory keyed on hashable statics (`_eta_jets_step_uniform` / `_ragged`,
# `_tt_variation_jets_step(n_probe, sum_over_probes, is_uniform)`), not through an argument of the body.
def _shift_one_slot(
        x_jet,  # elm_shape=(order+1,)+...
        xnp,
):              # same shape.  Slot t holds what slot t-1 held; slot 0 holds zero
    return xnp.concatenate([xnp.zeros_like(x_jet[:1]), x_jet[:-1]], axis=0)


def _product_order_step(
        carry,  # (c, a shifted r slots, G): (order+1,)+W+C+(rb,) ; (order+1,)+W+C+(ra,) ; C+(ra,n,rb)
        xs,     # (b[r],): W+C+(n,)
):              # (next carry, ())
    '''One coefficient r of b: c += (G . b[r]) . a_shifted, then shift the carried jet.  Closure-free scan body.'''
    c, a_shifted, G = carry
    (b_r,) = xs
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))              # only xnp
    Gb_WCab = contract('Caib,WCi->WCab', G, b_r)                               # E: once per coefficient of b
    c = c + contract('tWCa,WCab->tWCb', a_shifted, Gb_WCab)                    # c: against every slot carried
    return (c, _shift_one_slot(a_shifted, xnp), G), ()


def product_by_carried_shift(
        a,           # shape=(order+1,)+W+C+(ra,)
        G,           # shape=C+(ra,n,rb)
        b,           # shape=(order+1,)+W+C+(n,)
        is_uniform,  # bool.  The layout of the caller: decides what xscan is
):                   # c, shape=(order+1,)+W+C+(rb,)
    xnp, _, xscan = get_backend(is_uniform, tree_contains_jax((a, G, b)))
    c0 = xnp.zeros(a.shape[:-1] + (G.shape[-1],), xnp.result_type(a, G, b))
    (c, _, _), _ = xscan(_product_order_step, (c0, a, G), (b,))
    return c


def product_by_loop(
        a,  # shape=(order+1,)+W+C+(ra,)
        G,  # shape=C+(ra,n,rb)
        b,  # shape=(order+1,)+W+C+(n,)
):          # c, shape=(order+1,)+W+C+(rb,)
    xnp, _, _ = get_backend(True, tree_contains_jax((a, G, b)))                # only xnp
    P = a.shape[0]
    c = xnp.zeros(a.shape[:-1] + (G.shape[-1],), xnp.result_type(a, G, b))
    for r in range(P):
        Gb_WCab = contract('Caib,WCi->WCab', G, b[r])                          # E
        part = contract('tWCa,WCab->tWCb', a[:P - r], Gb_WCab)                 # c: only the P - r slots in range
        c = c + xnp.concatenate([xnp.zeros_like(c[:r]), part], axis=0)         # into slots r .. P-1
    return c


def product_of_full_jets(a, G, b, is_uniform, over_orders='scan'):
    """The option of 2.9, set by the plan.  The spelling of the setting is a placeholder."""
    if over_orders == 'scan':
        return product_by_carried_shift(a, G, b, is_uniform)
    if over_orders == 'loop':
        return product_by_loop(a, G, b)
    raise ValueError("over_orders must be 'scan' or 'loop', got %r" % (over_orders,))


# ---- the withdrawn third form, jax only, kept as the record of the comparison ------------------------------
def pair_table(P):
    """The pairs of orders (i of a, r of b) that a truncated product needs, in chunks of P, as static numpy tables."""
    pairs = [(i, r, i + r) for r in range(P) for i in range(P - r)]
    n_chunks = -(-len(pairs) // P)
    pad = n_chunks * P - len(pairs)
    table = np.array(pairs + [(0, 0, 0)] * pad).reshape((n_chunks, P, 3))
    valid = np.array([True] * len(pairs) + [False] * pad).reshape((n_chunks, P))
    return table[..., 0], table[..., 1], table[..., 2], valid, len(pairs), pad


def product_by_pair_scan(a, G, b):
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


def main(repo):
    common = load_library(repo)
    by_shift, by_loop = product_by_carried_shift, product_by_loop
    rng = np.random.default_rng(0)
    W, ra, n, rb = 5, 3, 4, 2
    G = rng.standard_normal((ra, n, rb))
    orders = (1, 2, 4, 8, 16)
    jets = {order: (rng.standard_normal((order + 1, W, ra)), rng.standard_normal((order + 1, W, n))) for order in orders}

    forms = (("'loop'", by_loop),
             ("'scan', uniform", lambda a, G, b: by_shift(a, G, b, True)),
             ("'scan', ragged", lambda a, G, b: by_shift(a, G, b, False)),
             ('pair table', product_by_pair_scan))
    print('1. Each form against the explicit sum, under jit (jax), and the lines of its traced program (scan body included)')
    print(('   %-6s' + ' %-22s' * len(forms)) % (('order',) + tuple(name for name, _ in forms)))
    for order in orders:
        a, b = jets[order]
        twin = J.mu_term(a, G, b)                                       # numpy, explicit sum over the orders
        ops = tuple(jnp.asarray(x) for x in (a, G, b))
        cells = []
        for name, f in forms:
            out = np.asarray(jax.jit(f)(*ops))
            ok = np.linalg.norm(out - twin) / np.linalg.norm(twin) < 1e-13
            cells.append('%s, %d lines' % ('correct' if ok else 'WRONG', program_size(f, *ops)))
        print(('   %-6d' + ' %-22s' * len(forms)) % ((order,) + tuple(cells)))

    print('2. Multiplications, relative to the loop')
    for label, sizes in (('W = 1000, bonds 64, mode rank 32', (1000, 64, 32, 64)), ('W = 1000, bonds 4, mode rank 4', (1000, 4, 4, 4))):
        print('   %s' % label)
        print('   %-6s %-16s %-16s' % ('order', "'scan'", 'pair table'))
        for order in orders:
            loop, shift, table = counts(order + 1, *sizes)
            print('   %-6d %-16s %-16s' % (order, '%.2f' % (shift / loop), '%.2f' % (table / loop)))

    print("3. The option on the four backends: over_orders='scan' (the default) against 'loop', orders 1 to 16")
    print('   %-16s %-10s %-14s %-20s %-16s' % ('inputs', 'layout', 'xscan is', 'the settings agree', 'output'))
    for kind, to_backend, array_type in (('numpy', np.asarray, np.ndarray), ('jax', jnp.asarray, jax.Array)):
        for layout, is_uniform in (('ragged', False), ('uniform', True)):
            _, _, xscan = common.get_backend(is_uniform, kind == 'jax')
            for dtype, tol in ((np.float64, 1e-13), (np.float32, 1e-5)):
                agree, typed = True, True
                for order in orders:
                    a, b = jets[order]
                    ops = tuple(to_backend(x.astype(dtype)) for x in (a, G, b))
                    scan = product_of_full_jets(*ops, is_uniform, over_orders='scan')
                    loop = product_of_full_jets(*ops, is_uniform, over_orders='loop')
                    typed &= all(isinstance(x, array_type) and x.dtype == dtype for x in (scan, loop))
                    twin = J.mu_term(a, G, b)
                    for out in (scan, loop):
                        agree &= bool(np.linalg.norm(np.asarray(out) - twin) / np.linalg.norm(twin) < tol)
                    agree &= bool(np.linalg.norm(np.asarray(scan) - np.asarray(loop)) / np.linalg.norm(twin) < tol)
                name = '%s %s' % (kind, np.dtype(dtype).name)
                print('   %-16s %-10s %-14s %-20s %-16s' % (
                    name, layout, xscan.__name__, 'to rounding' if agree else 'DIFFER', name if typed else 'WRONG TYPE'))


if __name__ == '__main__':
    main(sys.argv[1])
