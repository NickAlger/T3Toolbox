import sys
import numpy as np
sys.path.append('/home/claude/repo/T3Toolbox')
import t3toolbox.backend.fv_conversions as fvc, t3toolbox.backend.t3_constructors as t3c
np.random.seed(0)
for tt in ((1, 5, 6, 4, 1), (2, 5, 6, 4, 3)):
    try:
        x = t3c.t3_corewise_randn((12, 11, 13, 10), (4, 5, 4, 3), tt)
        frame, (Vc, Hc) = fvc.t3_orthogonal_representations(x)
    except Exception as e:
        print(tt, 'orthogonal representations failed:', type(e).__name__, str(e)[:100]); continue
    D, U, L, R = frame; d = len(D)
    S = [None] * (d + 1); S[d] = np.eye(L[-1].shape[2])
    for i in reversed(range(d)):
        S[i] = np.einsum('aib,bc,eic->ae', L[i], S[i + 1], R[i])
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    Hc2 = [np.einsum('aib,bc->aic', L[i], S[i + 1]) for i in range(d)]
    Vc2 = [np.einsum('aub,aib->ui', U[i], Hc2[i]) @ D[i] for i in range(d)]
    print('tt ranks', tt, 'frame ranks rL', [l.shape[0] for l in L] + [L[-1].shape[2]], 'rR', [r.shape[0] for r in R] + [R[-1].shape[2]])
    print('   TT centers rebuilt from (L, R) alone:     ', ' '.join('%.0e' % rel(a, b) if a.shape == b.shape else 'shape' for a, b in zip(Hc2, Hc)))
    print('   Tucker centers rebuilt from (D, U, L, R): ', ' '.join('%.0e' % rel(a, b) if a.shape == b.shape else 'shape' for a, b in zip(Vc2, Vc)))
