"""The dtype rule of jets._metric (K1 part 2's prose, finding 1): the result's dtype is the lowest floating precision
of r and every weight given; r and each weight are cast to it; each square is formed in it. Checked against today's
data_metric on the four float32/float64 pairings (one weight, as full and as block), and the two-weight case."""
import itertools
import numpy as np
import t3toolbox.backend.jets as jets

def lowest(*arrays):
    floats = [np.asarray(a).dtype for a in arrays if a is not None and np.issubdtype(np.asarray(a).dtype, np.floating)]
    return min(floats, key=lambda t: np.finfo(t).bits) if floats else np.dtype(float)

def metric(r, block, full, gamma):              # the proposed rule; block left-aligned on r's leading axes
    dt = lowest(r, block, full)
    r = np.asarray(r, dt)
    sq = None
    if block is not None:
        b = np.asarray(block, dt)
        sq = np.reshape(b * b, b.shape + (1,) * (r.ndim - b.ndim))
    if full is not None:
        f = np.asarray(full, dt)
        sq = f * f if sq is None else sq * (f * f)
    out = r if sq is None else sq * r
    return out if gamma is None else np.asarray(gamma, dt) * out

rng = np.random.default_rng(0)
r64 = rng.standard_normal((3, 5, 4))
b64 = 0.5 + rng.random(3)
for rd, wd in itertools.product((np.float32, np.float64), repeat=2):
    r, b = r64.astype(rd), b64.astype(wd)
    full = np.broadcast_to(b.reshape(3, 1, 1), r.shape).astype(wd)
    old_full = jets.data_metric(r, full, 2.0)
    old_block = jets.data_metric(r, b.reshape(3, 1, 1), 2.0)          # blocks_to_weight's placement, apply-like
    new_full, new_block = metric(r, None, full, 2.0), metric(r, b, None, 2.0)
    print('r %s, weight %s: today %s / %s; rule %s / %s; bitwise %s %s' % (
        np.dtype(rd).name, np.dtype(wd).name, old_full.dtype, old_block.dtype, new_full.dtype, new_block.dtype,
        np.array_equal(old_full, new_full) and old_full.dtype == new_full.dtype,
        np.array_equal(old_block, new_block) and old_block.dtype == new_block.dtype))
for bd, fd in ((np.float32, np.float64), (np.float64, np.float32)):
    out = metric(r64, b64.astype(bd), np.ones(r64.shape, fd), None)
    print('two weights, r float64, block %s, full %s: %s' % (np.dtype(bd).name, np.dtype(fd).name, out.dtype))
