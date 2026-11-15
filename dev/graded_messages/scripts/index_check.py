"""Which bond do the library's mus[i] and nus[i] live on? Reads it off the trailing dimensions and
checks the same-edge pairing <mus[i], nus[i-1]> against t3_apply. Usage: python3 index_check.py <repo>"""
import sys
import numpy as np
sys.path.append(sys.argv[1])
import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
rng = np.random.default_rng(0); rn = rng.standard_normal
Ns, n, r = (6, 5, 7, 4), (3, 4, 2, 3), (1, 2, 3, 4, 1)      # distinct interior bond ranks r1=2, r2=3, r3=4
d, W = 4, 5
B = [rn((n[i], Ns[i])) for i in range(d)]
G = [rn((r[i], n[i], r[i + 1])) for i in range(d)]
ww = [rn((W, N)) for N in Ns]
xis = probing.compute_xi(B, ww)
mus = probing.compute_mu(G, xis)
nus = probing.compute_nu(G, xis)
val = apply.t3_apply((B, G), ww)
print('TT ranks (bond 0..d):', r)
for i in range(d):
    print('  i=%d   mus[i] trailing dim %d  (bond %d)     nus[i] trailing dim %d  (bond %d)'
          % (i, mus[i].shape[-1], i, nus[i].shape[-1], i + 1))
print('pairing against t3_apply (relative error):')
for i in range(1, d):
    print('  <mus[%d], nus[%d]>  %.1e' % (i, i - 1, np.linalg.norm(np.einsum('wa,wa->w', mus[i], nus[i - 1]) - val) / np.linalg.norm(val)))
