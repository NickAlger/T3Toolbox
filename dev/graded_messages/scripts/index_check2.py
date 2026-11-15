"""Which bond does every left-swept and right-swept library variable live on (manifold frame with distinct
rL and rR ranks)? Usage: python3 index_check2.py <repo>"""
import sys
import numpy as np
sys.path.append(sys.argv[1])
import t3toolbox.backend.probing as p
rng = np.random.default_rng(0); rn = rng.standard_normal
Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
d, W = 4, 5
D = [rn((nD[i], Ns[i])) for i in range(d)]; U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]; R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
V = [rn((nU[i], Ns[i])) for i in range(d)]; H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
ww = [rn((W, N)) for N in Ns]; zt = [rn((W, N)) for N in Ns]; c = rn((W,))
xis = p.compute_xi(D, ww); mus = p.compute_mu(L, xis); nus = p.compute_nu(R, xis)
dxis = p.compute_dxi(V, ww)
sig = p.compute_sigma(H, R, U, xis, dxis, mus); tau = p.compute_tau(H, L, U, xis, dxis, nus)
dt = p.compute_deta_tilde(D, zt)
tt = p.compute_tau_tilde(dt, L, xis, mus); st = p.compute_sigma_tilde(dt, R, xis, nus)
sh = p.compute_sigma_hat(R, xis, c)
print('rL (bond 0..d) =', rL, '   rR (bond 0..d) =', rR)
def where(dims, ranks):
    for off in (0, 1):
        if all(dims[i] == ranks[i + off] for i in range(d)): return 'bond i' if off == 0 else 'bond i+1'
    return '??'
for name, xs, ranks, lab in (('mus', mus, rL, 'rL'), ('sigmas', sig, rR, 'rR'), ('tau_tildes', tt, rL, 'rL'),
                             ('nus', nus, rR, 'rR'), ('taus', tau, rL, 'rL'), ('sigma_tildes', st, rR, 'rR'),
                             ('sigma_hats', sh, rR, 'rR')):
    dims = [x.shape[-1] for x in xs]
    print('  %-13s dims over i = %-14s -> %s of %s' % (name, dims, where(dims, ranks), lab))
