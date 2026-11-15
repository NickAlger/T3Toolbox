import sys
import numpy as np
sys.path.append('/home/claude/repo/T3Toolbox')
import t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.t3_constructors as t3c, t3toolbox.backend.apply as apply, t3toolbox.backend.probing as probing
np.random.seed(0); rng = np.random.default_rng(0)
shape, tk, tt = (12, 11, 13, 10), (4, 5, 4, 3), (2, 5, 6, 4, 3)
x = t3c.t3_corewise_randn(shape, tk, tt)
W = 7; w = [rng.standard_normal((W, N)) for N in shape]
for squash in (True, False):
    frame, (Vc, Hc) = fvc.t3_orthogonal_representations(x, squash_tails=squash)
    D, U, L, R = frame
    a0 = apply.t3_apply(x, w)
    print('squash_tails=%s: L tails (%d, %d)  R tails (%d, %d)  apply (D,L) %.0e (D,R) %.0e' % (
        squash, L[0].shape[0], L[-1].shape[2], R[0].shape[0], R[-1].shape[2],
        np.linalg.norm(a0 - apply.t3_apply((D, L), w)) / np.linalg.norm(a0), np.linalg.norm(a0 - apply.t3_apply((D, R), w)) / np.linalg.norm(a0)))
    # probe value read off the frame with boundary vectors of the frame's own tail sizes
    bL, bR = np.ones((W, L[0].shape[0])), np.ones((W, R[-1].shape[2]))
    xi = [np.einsum('io,wo->wi', D[i], w[i]) for i in range(4)]
    mu = [bL]
    for i in range(4): mu.append(np.einsum('wa,aib,wi->wb', mu[i], L[i], xi[i]))
    nu = [None] * 4 + [bR]
    for i in reversed(range(4)): nu[i] = np.einsum('aib,wi,wb->wa', R[i], xi[i], nu[i + 1])
    z = [np.einsum('wa,aib,wb,io->wo', mu[i], U[i], nu[i + 1], Vc[i]) for i in range(4)]
    zr = probing.t3_probe(w, x)
    print('     probe read off the frame with ones of the frame\'s tail sizes: %.0e' % max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(z, zr)))
