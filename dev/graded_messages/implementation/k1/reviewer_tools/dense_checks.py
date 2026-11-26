"""Reviewer 3's own dense checks of K1's mathematics (not the implementer's check_math.py).

Run from the repository root at a tree where the OLD jets functions exist (282b46a9):
    PYTHONPATH=$PWD python -B dev/graded_messages/implementation/k1/reviewer_tools/dense_checks.py

Every reference value is an explicit loop over every index (np.ndindex), reading a size-1 axis at 0; no
broadcasting in the oracle. Sizes are chosen to coincide (d = n + 1 = |W| = |K| = |C| = N_i) so that an axis placed
on the wrong side cannot fail by shape.
"""
import itertools, math
import numpy as np


def at(a, idx):
    """a[idx] with every size-1 axis of a read at 0 (the meaning of a size of 1 on a stack axis)."""
    return a[tuple(0 if s == 1 else j for s, j in zip(a.shape, idx))]


def metric_loop_probe(order, r, b, f, gamma):
    m = len(order)
    out = []
    for i, r_i in enumerate(r):
        o = np.empty_like(r_i)
        for idx in np.ndindex(r_i.shape):
            bb = 1.0 if b is None else b[(i,) + idx[:m]]
            ff = 1.0 if f is None else at(f[i], idx)
            o[idx] = (1.0 if gamma is None else gamma) * bb ** 2 * ff ** 2 * r_i[idx]
        out.append(o)
    return out


def metric_loop_apply(order, r, b, f, gamma):
    m = len(order)
    o = np.empty_like(r)
    for idx in np.ndindex(r.shape):
        bb = 1.0 if b is None else b[idx[:m]]
        ff = 1.0 if f is None else at(f, idx)
        o[idx] = (1.0 if gamma is None else gamma) * bb ** 2 * ff ** 2 * r[idx]
    return o


def left(b_row, ndim):
    """The note's left alignment: a row of shape F placed on the leading axes of an array of rank ndim."""
    return np.reshape(b_row, b_row.shape + (1,) * (ndim - b_row.ndim))


def metric_note(r_X, b_X, f_X, gamma):
    """The note's _metric formula: gamma * ((b*b) * (f*f)) * r, b left-aligned (b_X already the row or table)."""
    w = None
    if b_X is not None:
        bl = left(np.asarray(b_X, dtype=r_X.dtype), r_X.ndim)
        w = bl * bl
    if f_X is not None:
        f_X = np.asarray(f_X, dtype=r_X.dtype)
        w = f_X * f_X if w is None else w * (f_X * f_X)
    out = r_X if w is None else w * r_X
    return out if gamma is None else np.asarray(gamma, dtype=r_X.dtype) * out


def report(name, ok):
    print(('PASS ' if ok else 'FAIL ') + name)
    return ok


def check1_left_alignment():
    rng = np.random.default_rng(1)
    n = 2; order = (n,); d = n + 1; F = (n + 1,); W = (3,); K = (3,); C = (3,); N = 3   # all sizes 3
    r = [rng.standard_normal(F + W + K + C + (N,)) for _ in range(d)]
    b = rng.standard_normal((d,) + F)
    f = [rng.standard_normal(F + (1,) + (1,) + C + (N,)) for _ in range(d)]      # W of 1, K placed (1), C full
    gamma = 0.7
    want = metric_loop_probe(order, r, b, f, gamma)
    got = [metric_note(r[i], b[i], f[i], gamma) for i in range(d)]
    ok = report('probe tuple, left-aligned rows, coinciding sizes (with K)', all(np.allclose(a, c, rtol=1e-14, atol=0) for a, c in zip(want, got)))
    packed = np.stack(r)
    got_p = metric_note(packed, b, np.stack(f), gamma)
    ok &= report('probe packed (d,)+F+W+K+C+(N,), the table (d,)+F left-aligned: bitwise the tuple',
                 all(np.array_equal(got_p[i], got[i]) for i in range(d)))
    # the trap: numpy's right alignment of a row of shape F against r[i] -- runs, wrong
    wrong = [b[i] ** 2 * r[i] for i in range(d)]
    ok &= report('right-aligned row (numpy default) runs and is WRONG on coinciding sizes',
                 not all(np.allclose(w_, metric_loop_probe(order, r, b, None, None)[i]) for i, w_ in enumerate(wrong)))
    # E3b: a per-slot vector (n+1,) passed for probe at d = n+1 -- kind-free prefix rule would accept it per MODE
    per_slot = rng.standard_normal(F)
    as_mode = [left(per_slot[i:i + 1], 1) ** 2 * r[i] for i in range(d)]          # (d,) read as a mode table
    as_slot = metric_loop_apply(order, r[0], per_slot, None, None)
    ok &= report('E3b: a (n+1,) per-slot table read as per-mode gives a different number (so the exact check is needed)',
                 not np.allclose(as_mode[0], as_slot))
    # apply: F + W + K + C, F == W == K == C in size
    ra = rng.standard_normal(F + W + K + C)
    ba = rng.standard_normal(F)
    fa = rng.standard_normal(F + W + (1,) + (1,))
    ok &= report('apply F+W+K+C, block F left-aligned, full W, K and C of 1',
                 np.allclose(metric_loop_apply(order, ra, ba, fa, gamma), metric_note(ra, ba, fa, gamma), rtol=1e-14, atol=0))
    ok &= report('apply: right-aligned block (numpy default) runs and is WRONG',
                 not np.allclose(ba ** 2 * ra, metric_loop_apply(order, ra, ba, None, None)))
    # two jet variables: order (1, 2), F = (2, 3); W = (2,), C = (3,): F's sizes reappear on W and C
    order2 = (1, 2); F2 = (2, 3)
    r2 = [rng.standard_normal(F2 + (2,) + (3,) + (N,)) for _ in range(2)]
    b2 = rng.standard_normal((2,) + F2)
    want2 = metric_loop_probe(order2, r2, b2, None, None)
    got2 = [metric_note(r2[i], b2[i], None, None) for i in range(2)]
    ok &= report('probe at two jet variables, F = (2, 3) equal to (W, C) in size', all(np.allclose(a, c, rtol=1e-14, atol=0) for a, c in zip(want2, got2)))
    ok &= report('  ... and the transposed table (b[i].T) gives another number', not np.allclose(
        metric_note(r2[0], b2[0].T.reshape(F2) if False else np.swapaxes(b2[0].reshape(2, 3), 0, 1).reshape(F2), None, None), got2[0]))
    entries_r = rng.standard_normal((3, 3)); entries_b = np.array(1.7)
    ok &= report('entries: block () is a number', np.allclose(metric_note(entries_r, entries_b, None, 2.0), 2.0 * 1.7 ** 2 * entries_r))
    return ok


def check2_bitwise_against_today():
    """The note's check 5, independently: every single-weight case bitwise today's jets.data_metric."""
    import t3toolbox.backend.jets as jets
    ok = True
    rng = np.random.default_rng(2)
    for dt_r, dt_b in itertools.product((np.float64, np.float32), (np.float64, np.float32)):
        order = (2,); F = (3,); d = 3; W = (3,); C = (3,)
        y = [rng.standard_normal(F + W + C + (N,)).astype(dt_r) for N in (3, 4, 5)]
        b = rng.standard_normal((d,) + F).astype(dt_b)
        old = jets.data_metric(y, jets.blocks_to_weight('probe', order, b, y), 0.3)
        dt = jets._lowest_precision_dtype(np, y[0], b)
        new = [metric_note(y[i].astype(dt), np.asarray(b[i], dtype=dt), None, 0.3) for i in range(d)]
        ok &= report('probe block only, r %s, b %s: bitwise today (dtype %s)' % (np.dtype(dt_r).name, np.dtype(dt_b).name, old[0].dtype),
                     all(o.dtype == n_.dtype and o.tobytes() == n_.tobytes() for o, n_ in zip(old, new)))
        f = [rng.standard_normal(F + (1,) + C + (N,)).astype(dt_b) for N in (3, 4, 5)]
        old = jets.data_metric(y, f, None)
        new = [metric_note(y[i].astype(dt), None, f[i], None) for i in range(d)]
        ok &= report('probe full only (W of 1): bitwise today', all(o.tobytes() == n_.tobytes() for o, n_ in zip(old, new)))
        ya = rng.standard_normal(F + W + C).astype(dt_r)
        ba = rng.standard_normal(F).astype(dt_b)
        old = jets.data_metric(ya, jets.blocks_to_weight('apply', order, ba, ya), 2.5)
        new = metric_note(ya.astype(dt), np.asarray(ba, dtype=dt), None, 2.5)
        ok &= report('apply block only: bitwise today', old.tobytes() == new.tobytes())
        # the part-2 bridge: today's apply blocks_to_weight output passed as a FULL weight
        fb = jets.blocks_to_weight('apply', order, ba, ya)
        new = metric_note(ya.astype(dt), None, fb, 2.5)
        ok &= report('bridge: apply blocks_to_weight output read as a full weight: bitwise today', old.tobytes() == new.tobytes())
        ye = rng.standard_normal(W + C).astype(dt_r)
        old = jets.data_metric(ye, jets.blocks_to_weight('entries', (), np.asarray(1.3, dtype=dt_b), ye), None)
        new = metric_note(ye.astype(dt), None, jets.blocks_to_weight('entries', (), np.asarray(1.3, dtype=dt_b), ye), None)
        ok &= report('bridge: entries blocks_to_weight output read as a full weight: bitwise today', old.tobytes() == new.tobytes())
    # both weights: rounding only
    y = [rng.standard_normal((3, 3, 3, 4))]
    b = rng.standard_normal((1, 3)); f = [rng.standard_normal((3, 1, 3, 4))]
    old = jets.data_metric(y, [jets.blocks_to_weight('probe', (2,), b, y)[0] * f[0]], None)[0]
    new = metric_note(y[0], b[0], f[0], None)
    rel = float(np.max(np.abs(old - new) / np.abs(old)))
    ok &= report('both weights: (b f)^2 r vs b^2 f^2 r to rounding (max rel %.1e)' % rel, rel < 1e-14)
    return ok


def check3_factorials():
    ok = True
    rng = np.random.default_rng(3)
    for order in ((3,), (2, 1)):
        m = len(order); F = tuple(n + 1 for n in order); d = 2; W = (4,)
        kfac = np.ones(F)
        for k in np.ndindex(F):
            kfac[k] = math.prod(math.factorial(t) for t in k)
        z_t = [rng.standard_normal(F + W + (N,)) for N in (3, 5)]         # model, Taylor
        y_t = [rng.standard_normal(F + W + (N,)) for N in (3, 5)]         # data, Taylor
        to_d = lambda a: a * left(kfac, a.ndim)                           # derivative = k! * Taylor, slot by slot
        b_d = rng.uniform(0.5, 2, (d,) + F)                                # chosen for derivative residuals
        f_d = [rng.uniform(0.5, 2, F + W + (N,)) for N in (3, 5)]          # per entry, chosen for derivatives
        mis_d = sum(float(np.sum(metric_loop_probe(order, [to_d(z) - to_d(y) for z, y in zip(z_t, y_t)], b_d, f_d, None)[i]
                                 * (to_d(z_t[i]) - to_d(y_t[i])))) for i in range(d))
        b_t = b_d * kfac[None]                                             # the k! into the block weight only, once
        mis_t = sum(float(np.sum(metric_loop_probe(order, [z - y for z, y in zip(z_t, y_t)], b_t, f_d, None)[i]
                                 * (z_t[i] - y_t[i]))) for i in range(d))
        ok &= report('order %s: k!-scaled block weight, full weight unchanged, same misfit (%.6e vs %.6e)' % (order, mis_d, mis_t),
                     abs(mis_d - mis_t) <= 1e-12 * abs(mis_d))
        wrong = sum(float(np.sum(metric_loop_probe(order, [z - y for z, y in zip(z_t, y_t)], b_d * kfac[None], [f * left(kfac, f.ndim) for f in f_d], None)[i]
                                  * (z_t[i] - y_t[i]))) for i in range(d))
        ok &= report('order %s: k! in BOTH weights (the double count) differs' % (order,), abs(wrong - mis_d) > 1e-6 * abs(mis_d))
        # today's functions: derivative_weight_to_taylor_weight(blocks_to_weight(b)) == blocks_to_weight(k! b), bitwise
        import t3toolbox.backend.jets as jets
        old = jets.derivative_weight_to_taylor_weight('probe', order, jets.blocks_to_weight('probe', order, b_d, y_t))
        new = jets.blocks_to_weight('probe', order, b_d * kfac[None], y_t)
        ok &= report('order %s: today k! on the placed weight == k! on the table, bitwise' % (order,), old.tobytes() == new.tobytes())
        y32 = [a.astype(np.float32) for a in y_t]
        _, w_none, _ = jets.prepare_derivative_data('probe', order, y32)
        table = np.broadcast_to(kfac.astype(np.float32), (d,) + F)
        ok &= report('order %s: prepare_derivative_data(weight=None) is the k! table in y\'s dtype (float32), bitwise' % (order,),
                     w_none.dtype == np.float32 and w_none.reshape((d,) + F).tobytes() == np.ascontiguousarray(table).tobytes())
    return ok


def check4_y_metric_y():
    ok = True
    rng = np.random.default_rng(4)
    order = (1,); F = (2,); W = (3,); C = (3,); Ns = (3, 3)              # |C| = |W| = N_i
    y = [rng.standard_normal(F + W + (1,) + (N,)) for N in Ns]            # y shared by the stack: C of 1
    b = rng.uniform(0.5, 2, (2,) + F)
    f = [rng.uniform(0.5, 2, F + (1,) + C + (N,)) for N in Ns]             # full weight: W of 1, C full
    # loop over z's shape F+W+C+(N,), reading size-1 axes at 0
    total = np.zeros(C); blocks = np.zeros((2,) + F)
    for i, N in enumerate(Ns):
        for idx in np.ndindex(F + W + C + (N,)):
            v = at(y[i], idx) ** 2 * b[(i,) + idx[:1]] ** 2 * at(f[i], idx) ** 2
            total[idx[2]] += v; blocks[(i,) + idx[:1]] += v
    # per element of C: that element's own fit with y and f read at the element
    own = np.array([sum(float(np.sum(y[i][..., 0, :] ** 2 * left(b[i], 3) ** 2 * f[i][..., c, :] ** 2)) for i in range(2))
                    for c in range(3)])
    ok &= report('<y, M y> per element of C (y C of 1) = each element\'s own fit', np.allclose(total, own, rtol=1e-14))
    # the note's recipe: broadcast y to z's shape first, then product, then reduce
    recipe_b = np.stack([np.sum(np.broadcast_to(y[i], F + W + C + (N,)) * metric_note(np.broadcast_to(y[i], F + W + C + (N,)), b[i], f[i], None),
                                axis=(1, 2, 3)) for i, N in enumerate(Ns)])
    ok &= report('y_block_sumsq: broadcast-first recipe = loop', np.allclose(recipe_b, blocks, rtol=1e-14))
    # with a full weight whose C is 1 too, reducing WITHOUT the broadcast undercounts by |C|
    f1 = [rng.uniform(0.5, 2, F + (1,) + (1,) + (N,)) for N in Ns]
    no_bc = np.stack([np.sum(y[i] * metric_note(y[i], b[i], f1[i], None), axis=(1, 2, 3)) for i in range(2)])
    bc = np.stack([np.sum(np.broadcast_to(y[i], F + W + C + (N,)) * metric_note(np.broadcast_to(y[i], F + W + C + (N,)), b[i], f1[i], None),
                          axis=(1, 2, 3)) for i, N in enumerate(Ns)])
    ok &= report('without the broadcast the block sums are |C| times too small', np.allclose(bc, 3 * no_bc, rtol=1e-14))
    # today's oracle helper with the product weight: same number (the old full-form weight b*f)
    import t3toolbox.backend.fitting as bfit, t3toolbox.backend.jets as jets
    omega = [jets.blocks_to_weight('probe', order, b, y)[i] * f[i] for i in range(2)]
    old = bfit._y_block_metric_y('probe', order, y, omega, W, C, Ns)
    ok &= report('today\'s _y_block_metric_y with omega = b f agrees to rounding', np.allclose(old, blocks, rtol=1e-13))
    return ok


if __name__ == '__main__':
    results = [check1_left_alignment(), check2_bitwise_against_today(), check3_factorials(), check4_y_metric_y()]
    print('ALL PASS' if all(results) else 'SOME FAILED')
