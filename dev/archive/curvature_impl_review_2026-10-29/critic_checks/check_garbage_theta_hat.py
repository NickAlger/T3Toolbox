"""Critic check (read-only): in the UNIFORM layout (boundary bonds padded too), does garbage in the padded slots of
the core DIRECTION theta_hat (clean theta, clean samples) contaminate the REAL block of the probe_derivatives
two-hole?  If yes, theta_hat must be masked on entry inside the uniform kind's corewise_hessian."""
import sys; sys.path.insert(0, __import__("os").getcwd())
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "checks"))
import numpy as np
import t3toolbox.backend.t3_constructors as t3k
from c14_probe_derivs_twohole import two_hole
rng = np.random.default_rng(31); np.random.seed(31)
shape, tr, ttr, order, Wn, K = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1), 2, 5, (2,)
Np, n, r = 7, 4, 3
U, G = t3k.t3_corewise_randn(shape, tr, ttr)
Ud = tuple(rng.standard_normal(K + u.shape) for u in U); Gd = tuple(rng.standard_normal(K + g.shape) for g in G)
ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
zt = [rng.standard_normal((order + 1, Wn, N)) for N in shape]
ref = two_hole(U, G, Ud, Gd, ww, pp, zt, order, K, ())
def padto(a, tgt, fill=0.0):
    out = np.full(a.shape[:a.ndim - len(tgt)] + tgt, fill); out[tuple(slice(0, s) for s in a.shape)] = a; return out
def padg(a, tgt):          # garbage in the padded region
    out = 1e3 * rng.standard_normal(a.shape[:a.ndim - len(tgt)] + tgt); out[tuple(slice(0, s) for s in a.shape)] = a; return out
Up = [padto(u, (n, Np)) for u in U]; Gp = [padto(g, (r, n, r)) for g in G]
wwp = [padto(w, (Np,)) for w in ww]; ppp = [padto(p, (Np,)) for p in pp]; ztp = [padto(z, (Np,)) for z in zt]
for label, Udp, Gdp in (('clean (zero-padded) theta_hat', [padto(u, (n, Np)) for u in Ud], [padto(g, (r, n, r)) for g in Gd]),
                        ('GARBAGE-padded theta_hat', [padg(u, (n, Np)) for u in Ud], [padg(g, (r, n, r)) for g in Gd])):
    out = two_hole(Up, Gp, Udp, Gdp, wwp, ppp, ztp, order, K, ())
    err = max(np.linalg.norm(a[tuple(slice(0, s) for s in b.shape)] - b) / np.linalg.norm(b)
              for a, b in zip(out[0] + out[1], ref[0] + ref[1]))
    print(f'uniform-layout padding, {label}: real-block rel err {err:.1e}')

# Garbage ONLY in the padded BOUNDARY-bond slots of theta_hat's TT cores (left bond of core 0, right bond of the
# last core) -- exactly where the existing uniform COREWISE GN hvp leaves garbage (measured 2.7e5).
Udp = [padto(u, (n, Np)) for u in Ud]
Gdp = [padto(g, (r, n, r)) for g in Gd]
Gdp[0][..., 1:, :, :] = 1e3 * rng.standard_normal(Gdp[0][..., 1:, :, :].shape)
Gdp[-1][..., :, :, 1:] = 1e3 * rng.standard_normal(Gdp[-1][..., :, :, 1:].shape)
out = two_hole(Up, Gp, Udp, Gdp, wwp, ppp, ztp, order, K, ())
err = max(np.linalg.norm(a[tuple(slice(0, s) for s in b.shape)] - b) / np.linalg.norm(b)
          for a, b in zip(out[0] + out[1], ref[0] + ref[1]))
print(f'uniform-layout padding, garbage ONLY in padded boundary-bond slots of theta_hat: real-block rel err {err:.1e}')
