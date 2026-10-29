"""Critic check (read-only): does the EXISTING one-hole probe_derivatives corewise transpose leak into padded
BOUNDARY-bond slots when the uniform layout pads r_0 / r_d (ones-initialized carries)? And which existing
uniform code masks the result? Compares with the two-hole leak of check_c16_boundary.py."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.t3_constructors as t3k
import t3toolbox.backend.sampling_derivatives as sd
rng = np.random.default_rng(31); np.random.seed(31)
shape, tr, ttr, order, Wn = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1), 2, 5
Np, n, r = 7, 4, 3
U, G = t3k.t3_corewise_randn(shape, tr, ttr)
ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
zt = [rng.standard_normal((order + 1, Wn, N)) for N in shape]
def padto(a, tgt):
    out = np.zeros(a.shape[:a.ndim - len(tgt)] + tgt); out[tuple(slice(0, s) for s in a.shape)] = a; return out
for label, rl, rr in (('interior-only padding', lambda i: 1 if i == 0 else r, lambda i: 1 if i == 2 else r),
                      ('boundary bonds padded (uniform layout)', lambda i: r, lambda i: r)):
    Up = [padto(u, (n, Np)) for u in U]; Gp = [padto(g, (rl(i), n, rr(i))) for i, g in enumerate(G)]
    wwp = [padto(w, (Np,)) for w in ww]; ppp = [padto(p, (Np,)) for p in pp]; ztp = [padto(z, (Np,)) for z in zt]
    ref = sd.t3_probe_corewise_derivatives_transpose(zt, ww, pp, (U, G), order, sum_over_probes=True, chunk_size=None)
    out = sd.t3_probe_corewise_derivatives_transpose(ztp, wwp, ppp, (tuple(Up), tuple(Gp)), order, sum_over_probes=True, chunk_size=None)
    err, leak = 0.0, 0.0
    for a, b in zip(out[0] + out[1], ref[0] + ref[1]):
        sl = tuple(slice(0, s) for s in b.shape)
        err = max(err, np.linalg.norm(a[sl] - b) / np.linalg.norm(b))
        m = np.ones(a.shape, bool); m[sl] = False
        leak = max(leak, np.abs(a[m]).max() if m.any() else 0.0)
    print(f'one-hole corewise transpose, {label}: real-block rel err {err:.1e}; max |padded output| {leak:.1e}')
