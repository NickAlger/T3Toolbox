"""H6 side-check (sharing): is shared(MANIFOLD).project(v) in the tangent space of the TIED set {x = B G B^T}? (d=2, sharing (0,0))
Compare against a finite-difference Jacobian of the tied parametrization."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.shared_geometry as sg
np.random.seed(3)
M = t3m.MANIFOLD
N, n, r = 4, 2, 2
x = t3.TuckerTensorTrain.randn((N, N), (n, n), (1, r, 1)).share((0, 0)); S = sg.shared_manifold((0, 0)); f = S.frame(x)
B = np.asarray(x.tucker_cores[0]); G0 = np.asarray(x.tt_cores[0]); G1 = np.asarray(x.tt_cores[1])   # B: (n, N); G0: (1, n, r); G1: (r, n, 1)
def dense(B, G0, G1):
    return np.einsum('ai,ear,rbf,bj->ij', B, G0, G1, B)    # (N, N)
x0 = dense(B, G0, G1); assert np.allclose(x0, x.to_dense())
cols = []
eps = 1e-6
for arr, name in [(B, 'B'), (G0, 'G0'), (G1, 'G1')]:
    for idx in np.ndindex(arr.shape):
        Ap = arr.copy(); Ap[idx] += eps
        cols.append(((dense(*[Ap if a is arr else a for a in (B, G0, G1)]) - x0) / eps).ravel())
J_tied = np.stack(cols, axis=1)             # (N*N, n*N + n*r + r*n)
# the FULL manifold parametrization B0 G B1^T
cols_full = []
for which in range(2):
    for idx in np.ndindex(B.shape):
        Bp = B.copy(); Bp[idx] += eps
        Bs = [B, B]; Bs[which] = Bp
        cols_full.append(((np.einsum('ai,ear,rbf,bj->ij', Bs[0], G0, G1, Bs[1]) - x0) / eps).ravel())
J_full = np.concatenate([np.stack(cols_full, axis=1), J_tied[:, n * N:]], axis=1)
print('rank tied-set Jacobian =', np.linalg.matrix_rank(J_tied, tol=1e-4), ' rank full-manifold Jacobian =', np.linalg.matrix_rank(J_full, tol=1e-4))
def resid(vec, J):
    c, *_ = np.linalg.lstsq(J, vec, rcond=None); return np.linalg.norm(J @ c - vec) / np.linalg.norm(vec)
for k in range(3):
    v = M.randn(f); pv = S.project(v)
    print('sample %d: raw gauged tangent: dist to tied-tangent-space=%.2e (full: %.2e) | S.project(v): dist to tied=%.2e, |pv - v|/|v|=%.2e'
          % (k, resid(v.to_dense().ravel(), J_tied), resid(v.to_dense().ravel(), J_full), resid(pv.to_dense().ravel(), J_tied),
             np.linalg.norm(pv.to_dense() - v.to_dense()) / np.linalg.norm(v.to_dense())))
# and the ambient projection route
g = np.random.randn(N, N)
pa = S.project_ambient(f, g)
print('S.project_ambient(dense g): dist to tied tangent space = %.2e ; dist to full tangent space = %.2e' % (resid(pa.to_dense().ravel(), J_tied), resid(pa.to_dense().ravel(), J_full)))
