import sys
import numpy as np
sys.path.append('/home/claude/repo/T3Toolbox')
import t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.t3_constructors as t3c, t3toolbox.backend.probing as probing, t3toolbox.backend.apply as apply
np.random.seed(0); rng = np.random.default_rng(0)
shape, tk, tt = (12, 11, 13, 10), (4, 5, 4, 3), (2, 5, 6, 4, 3)
x = t3c.t3_corewise_randn(shape, tk, tt)
B, G = x
print('point TT core shapes:', [g.shape for g in G])
frame, (Vc, Hc) = fvc.t3_orthogonal_representations(x)
D, U, L, R = frame
print('L shapes:', [a.shape for a in L]); print('R shapes:', [a.shape for a in R]); print('U shapes:', [a.shape for a in U])
W = 7
w = [rng.standard_normal((W, N)) for N in shape]
a0 = apply.t3_apply(x, w)
aL = apply.t3_apply((D, L), w); aR = apply.t3_apply((D, R), w)
print('apply of x vs (D,L): %.1e   vs (D,R): %.1e' % (np.linalg.norm(a0 - aL) / np.linalg.norm(a0), np.linalg.norm(a0 - aR) / np.linalg.norm(a0)))
