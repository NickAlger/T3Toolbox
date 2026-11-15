"""Pad the samples to a multiple of the block, reshape W -> (blocks, B), and loop over the leading axis.
Checks, with the library's own kernels (numpy): padded samples contribute exactly nothing where the sample
enters multilinearly (apply, probe), and what is needed where it does not (entries)."""
import sys
import numpy as np
sys.path.append(sys.argv[1])
import t3toolbox.backend.t3_constructors as t3c
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries

rng = np.random.default_rng(0); np.random.seed(0)
shape, W, B = (7, 6, 8, 5), 10, 4
x = t3c.t3_corewise_randn(shape, (3, 3, 3, 3), (1, 3, 3, 3, 1))
frame, _ = fvc.t3_orthogonal_representations(x)
V = [rng.standard_normal((u.shape[1], d.shape[1])) for u, d in zip(frame[1], frame[0])]
H = [rng.standard_normal((l.shape[0], l.shape[1], r.shape[2])) for l, r in zip(frame[2], frame[3])]
flat = lambda v: np.concatenate([np.ravel(a) for part in v for a in part])
rel = lambda a, b: np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b))
pad = (-W) % B
nb = (W + pad) // B
padw = lambda a: np.concatenate([a, np.zeros((pad,) + a.shape[1:], a.dtype)])          # zero rows
blocks = lambda a: padw(a).reshape((nb, B) + a.shape[1:])                                # W -> (blocks, B)

# ---- probe: zero sample vectors, zero data
w = [rng.standard_normal((W, N)) for N in shape]
y = [rng.standard_normal((W, N)) for N in shape]
r = [z - yy for z, yy in zip(probing.t3_probe(w, x), y)]
g_ref = probing.tv_probe_transpose(r, w, frame, sum_over_probes=True)
Jp = probing.tv_probe(w, (V, H), frame)
H_ref = probing.tv_probe_transpose(Jp, w, frame, sum_over_probes=True)
wb, yb = [blocks(a) for a in w], [blocks(a) for a in y]
g = Hp = None
f = 0.0
for k in range(nb):                                                  # the scan over the new leading axis
    wk, yk = [a[k] for a in wb], [a[k] for a in yb]
    rk = [z - yy for z, yy in zip(probing.t3_probe(wk, x), yk)]
    f += 0.5 * sum(float(np.sum(a ** 2)) for a in rk)
    gk = probing.tv_probe_transpose(rk, wk, frame, sum_over_probes=True)
    Hk = probing.tv_probe_transpose(probing.tv_probe(wk, (V, H), frame), wk, frame, sum_over_probes=True)
    g = gk if g is None else tuple(tuple(a + b for a, b in zip(p, q)) for p, q in zip(g, gk))
    Hp = Hk if Hp is None else tuple(tuple(a + b for a, b in zip(p, q)) for p, q in zip(Hp, Hk))
f_ref = 0.5 * sum(float(np.sum(a ** 2)) for a in r)
print('probe,   W=%d padded to %d, blocks of %d:  f %.0e   g %.0e   Hp %.0e' % (W, W + pad, B, abs(f - f_ref) / f_ref, rel(g, g_ref), rel(Hp, H_ref)))
zpad = probing.t3_probe([np.zeros((pad, N)) for N in shape], x)
print('         probe output at the padded samples: largest entry %.1e' % max(np.abs(z).max() for z in zpad))
# the library's kernels on the reshaped stack directly (W has two axes), no loop at all
g2 = probing.tv_probe_transpose([blocks(a) for a in r], wb, frame, sum_over_probes=True)
print('         the same kernels on the two-axis sample stack, no loop:  g %.0e' % rel(g2, g_ref))

# ---- apply: zero sample vectors
c = apply.t3_apply(x, w) - rng.standard_normal(W)
ga_ref = apply.tv_apply_transpose(c, w, frame, sum_over_probes=True)
cp = apply.t3_apply(x, [padw(a) for a in w])
print('apply:   value at the padded samples: largest entry %.1e' % np.abs(cp[W:]).max())

# ---- entries: an index cannot be zero; pad with any valid index
index = np.stack([rng.integers(0, N, size=W) for N in shape])
ip = np.concatenate([index, np.zeros((len(shape), pad), dtype=index.dtype)], axis=1)
ce = entries.tv_entries(index, (V, H), frame)                                           # J p at the real samples
cep = entries.tv_entries(ip, (V, H), frame)                                             # J p at real + padded
He_ref = entries.tv_entries_transpose(ce, index, frame, sum_over_probes=True)
He_bad = entries.tv_entries_transpose(cep, ip, frame, sum_over_probes=True)
mask = np.concatenate([np.ones(W), np.zeros(pad)])
He_ok = entries.tv_entries_transpose(mask * cep, ip, frame, sum_over_probes=True)
print('entries: J p at the padded samples: largest entry %.1e' % np.abs(cep[W:]).max())
print('         Hp with padded indices, no mask %.0e;  with a zero weight on the padded samples %.0e' % (rel(He_bad, He_ref), rel(He_ok, He_ref)))
