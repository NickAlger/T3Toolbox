"""Read-only plan check (v2): mask-on-entry suffices for the two-hole (the uniform equivalence contract on
real parts). Pad theta and theta_hat with ZEROS (what masking produces) to larger ranks and mode size,
run the probe_derivatives two-hole, and compare the real block with the unpadded result; also check the
padded block of the output is exactly 0 (no phantom rank leaks out of the multilinear map)."""
import sys; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, sys.path[0] if False else __import__('os').path.dirname(__file__))
import numpy as np
import t3toolbox.backend.t3_constructors as t3k
from c14_probe_derivs_twohole import two_hole
rng = np.random.default_rng(31); np.random.seed(31)
shape, tr, ttr, order, Wn, K = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1), 2, 5, (2,)
Np, n, r = 7, 4, 3          # padded mode size, Tucker rank, TT rank (uniform supercore sizes)
U, G = t3k.t3_corewise_randn(shape, tr, ttr)
Ud = tuple(rng.standard_normal(K + u.shape) for u in U); Gd = tuple(rng.standard_normal(K + g.shape) for g in G)
ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
zt = [rng.standard_normal((order + 1, Wn, N)) for N in shape]
ref = two_hole(U, G, Ud, Gd, ww, pp, zt, order, K, ())
def padto(a, tgt):
    out = np.zeros(a.shape[:a.ndim - len(tgt)] + tgt); out[tuple(slice(0, s) for s in a.shape)] = a; return out
d = len(shape)
rl = lambda i: 1 if i in (0,) else r; rr = lambda i: 1 if i == d - 1 else r
Up = [padto(u, (n, Np)) for u in U]; Gp = [padto(g, (rl(i), n, rr(i))) for i, g in enumerate(G)]
Udp = [padto(u, (n, Np)) for u in Ud]; Gdp = [padto(g, (rl(i), n, rr(i))) for i, g in enumerate(Gd)]
wwp = [padto(w, (Np,)) for w in ww]; ppp = [padto(p, (Np,)) for p in pp]; ztp = [padto(z, (Np,)) for z in zt]
out = two_hole(Up, Gp, Udp, Gdp, wwp, ppp, ztp, order, K, ())
err, leak = 0.0, 0.0
for a, b in zip(out[0] + out[1], ref[0] + ref[1]):
    sl = tuple(slice(0, s) for s in b.shape)
    err = max(err, np.linalg.norm(a[sl] - b) / np.linalg.norm(b))
    m = np.ones(a.shape, bool); m[sl] = False
    leak = max(leak, np.abs(a[m]).max() if m.any() else 0.0)
print('zero-padded inputs: real-block rel err %.1e ; max |padded output| %.1e' % (err, leak))
