#!/usr/bin/env python3
"""K1's mathematics, checked by small dense computations before the design leans on them (the design note, §3).

    PYTHONPATH=<repo> python dev/graded_messages/implementation/k1/check_math.py

Plain numpy with loops; the library is imported only for the comparisons marked "against today's tree" (the pre-K1
`jets.data_metric` / `blocks_to_weight` / `prepare_derivative_data`), never as an oracle of the new design.
Each check prints one line and asserts.
"""
import itertools
import math

import numpy as np

rng = np.random.default_rng(0)


def factorials(order):
    F = tuple(n + 1 for n in order)
    out = np.ones(F)
    for k in itertools.product(*(range(f) for f in F)):
        out[k] = math.prod(math.factorial(t) for t in k)
    return out


def left_aligned(block_F, array):
    '''The block weight's row on the leading axes of one array: F + (1,)*rest.'''
    return np.reshape(block_F, np.shape(block_F) + (1,) * (np.ndim(array) - np.ndim(block_F)))


# ---- 1. M = gamma * b**2 * f**2, entry by entry, against loops over the blocks --------------------------------------
d, order, W, K, C, Ns = 2, (2,), (4,), (3,), (2,), (5, 6)
F = tuple(n + 1 for n in order)
b = rng.random((d,) + F) + 0.5                                           # probe block weight (d,)+F
f = [rng.random(F + (1,) + (1,) + C + (N,)) + 0.5 for N in Ns]           # full weight, W of 1, K placed (1), C full
r = [rng.standard_normal(F + W + K + C + (N,)) for N in Ns]               # r with K: F+W+K+C+(Ni,)
gamma = 0.7
Mr = [gamma * left_aligned(b[i], r[i]) ** 2 * f[i] ** 2 * r[i] for i in range(d)]
for i in range(d):
    for idx in np.ndindex(r[i].shape):
        k = idx[:len(F)]
        fi = f[i][tuple(0 if s == 1 else j for s, j in zip(f[i].shape, idx))]
        assert np.isclose(Mr[i][idx], gamma * b[i][k] ** 2 * fi ** 2 * r[i][idx], rtol=1e-15)
print('1. M r = gamma b^2 f^2 r, the block row of mode i on its leading F axes (tuple r with K): checked by loop')

# packed r with K: (d,)+F+W+K+C+(N,), the block weight (d,)+F left-aligned once
N = max(Ns)
rp = np.stack([np.pad(r[i], ((0, 0),) * (r[i].ndim - 1) + ((0, N - Ns[i]),)) for i in range(d)])
fp = np.stack([np.pad(f[i], ((0, 0),) * (f[i].ndim - 1) + ((0, N - Ns[i]),)) for i in range(d)])
Mrp = gamma * left_aligned(b, rp) ** 2 * fp ** 2 * rp
for i in range(d):
    assert np.array_equal(Mrp[i][..., :Ns[i]], Mr[i])
    assert not np.any(Mrp[i][..., Ns[i]:])
print('2. packed r with K: the block weight (d,)+F on the leading axes is the tuple metric mode by mode (bitwise), '
      'zero padding kept')

# ---- 3. the k! of derivative coordinates goes into the block weight only, once ----------------------------------------
order = (2, 1)
F = tuple(n + 1 for n in order)
fac = factorials(order)
Wd = (3,)
y_der = [rng.standard_normal(F + Wd + (N,)) for N in Ns]                 # data as derivatives
z_tay = [rng.standard_normal(F + Wd + (N,)) for N in Ns]                 # a model, Taylor coefficients
b_d = rng.random((d,) + F) + 0.5                                         # a block weight chosen for derivatives
f_d = [rng.random((1,) * len(F) + Wd + (N,)) + 0.5 for N in Ns]         # a full weight in derivative coordinates
z_der = [left_aligned(fac, z) * z for z in z_tay]                        # Taylor -> derivatives: times k!
misfit_der = 0.5 * sum(np.sum((left_aligned(b_d[i], y_der[i]) * f_d[i] * (z_der[i] - y_der[i])) ** 2)
                       for i in range(d))
y_tay = [y / left_aligned(fac, y) for y in y_der]
b_t = b_d * fac                                                          # (k! b_d): the block weight only
misfit_tay = 0.5 * sum(np.sum((left_aligned(b_t[i], y_tay[i]) * f_d[i] * (z_tay[i] - y_tay[i])) ** 2)
                       for i in range(d))
assert np.isclose(misfit_der, misfit_tay, rtol=1e-13), (misfit_der, misfit_tay)
print('3. derivative data (b_d, f_d) = Taylor data (k! b_d, f_d): the k! into the block weight once, the full '
      'weight unchanged (%.6e = %.6e)' % (misfit_der, misfit_tay))
# with no weight: the Taylor block weight is the k! table itself
misfit_der0 = 0.5 * sum(np.sum((z_der[i] - y_der[i]) ** 2) for i in range(d))
misfit_tay0 = 0.5 * sum(np.sum((left_aligned(fac, y_tay[i]) * (z_tay[i] - y_tay[i])) ** 2) for i in range(d))
assert np.isclose(misfit_der0, misfit_tay0, rtol=1e-13)
print('   no weight: the Taylor block weight is the k! table, (d,)+F for probe (checked)')

# ---- 4. <y, M y> and the block sums over z's shape when y's C axes are 1 ----------------------------------------------
order, W, C = (1,), (5,), (3,)
F = (2,)
b = rng.random((d,) + F) + 0.5
y = [rng.standard_normal(F + W + (1,) + (N,)) for N in Ns]               # y shared by the stack: C axis of 1
fw = [rng.random(F + (1,) + (1,) + (N,)) + 0.5 for N in Ns]             # full weight: W and C of 1
# the oracle's sums, as one broadcast to z's shape and one reduction per mode
ymy_C = sum(np.sum(np.broadcast_to(left_aligned(b[i], y[i]) ** 2 * fw[i] ** 2 * y[i] ** 2, F + W + C + (Ns[i],)),
                   axis=(0, 1, 3)) for i in range(d))                    # keeps C: shape C
blocks = np.stack([np.sum(np.broadcast_to(left_aligned(b[i], y[i]) ** 2 * fw[i] ** 2 * y[i] ** 2,
                                          F + W + C + (Ns[i],)), axis=(1, 2, 3)) for i in range(d)])  # (d,)+F
# against: each element c of C fitted on its own (y and the weights at c, read at 0 where of size 1)
for c in range(C[0]):
    per = sum(np.sum((b[i][:, None, None] * fw[i][:, :, 0] * y[i][:, :, 0]) ** 2) for i in range(d))
    assert np.isclose(ymy_C[c], per, rtol=1e-14)
loop = np.zeros((d,) + F)
for i in range(d):
    for k in range(F[0]):
        for c in range(C[0]):
            loop[i, k] += np.sum((b[i, k] * fw[i][k, :, 0] * y[i][k, :, 0]) ** 2)
assert np.allclose(blocks, loop, rtol=1e-14)
print('4. <y, M y> per element of C, and the block sums counted |C| times, equal the loops over the elements when '
      "y's C axes are 1 (broadcast to z's shape first)")
# the trap the broadcast avoids: summing the size-1 axis as it stands undercounts the block sums by |C|
under = np.stack([np.sum(left_aligned(b[i], y[i]) ** 2 * fw[i] ** 2 * y[i] ** 2, axis=(1, 2, 3)) for i in range(d)])
assert np.allclose(under * C[0], blocks)
print('   (without the broadcast the block sums are |C| = %d times too small)' % C[0])

# ---- 5. the battery's arithmetic: bitwise for one weight, rounding when both are given --------------------------------
import t3toolbox.backend.jets as jets                                     # today's tree: the old metric
order, W = (2,), (6,)
F = (3,)
y = [rng.standard_normal(F + W + (N,)) for N in Ns]
r = [rng.standard_normal(F + W + (N,)) for N in Ns]
bb = rng.random((d,) + F) + 0.5
old_w = jets.blocks_to_weight('probe', order, bb, y)                     # old block weight, (d,)+F+(1,)+(1,)
old = jets.data_metric(r, old_w, None)
new = tuple((left_aligned(bb[i], r[i]) * left_aligned(bb[i], r[i])) * r[i] for i in range(d))   # (b*b)*r
assert all(np.array_equal(a, c) for a, c in zip(old, new))
full = [rng.random(F + W + (N,)) + 0.5 for N in Ns]
old = jets.data_metric(r, full, None)
new = tuple((full[i] * full[i]) * r[i] for i in range(d))
assert all(np.array_equal(a, c) for a, c in zip(old, new))
print('5. one weight: (b*b)*r for an old block weight and (f*f)*r for an old full weight are bitwise the old metric')
both_old = jets.data_metric(r, [left_aligned(bb[i], full[i]) * full[i] for i in range(d)], None)   # omega = b f
both_new = tuple((left_aligned(bb[i], r[i]) ** 2 * (full[i] * full[i])) * r[i] for i in range(d))
diff = max(float(np.max(np.abs(a - c) / np.abs(a))) for a, c in zip(both_old, both_new))
assert diff < 1e-15
print('   both weights: b^2 f^2 r against (b f)^2 r differ by rounding only (max rel %.1e)' % diff)

# ---- 6. the old prepare_derivative_data's weight against the new block table (bitwise) -------------------------------
order = (3,)
F = (4,)
y_der = [rng.standard_normal(F + (5,) + (N,)) for N in Ns]
_, w_old, _ = jets.prepare_derivative_data('probe', order, y_der)        # (d,)+F+(1,)+(1,): k! per mode and slot
assert np.array_equal(w_old[..., 0, 0], np.broadcast_to(factorials(order), (d,) + F))
blocks_d = rng.random((d,) + F) + 0.5
_, w_old, _ = jets.prepare_derivative_data('probe', order, y_der, jets.blocks_to_weight('probe', order, blocks_d,
                                                                                         y_der))
assert np.array_equal(w_old[..., 0, 0], blocks_d * factorials(order))
print('6. the old prepare_derivative_data weight is the new block weight k! * b_d, reshaped (bitwise)')
print('all checks pass')
