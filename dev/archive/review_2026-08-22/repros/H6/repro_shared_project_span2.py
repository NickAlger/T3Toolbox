"""Analytic tangent Jacobians for d=2: tied set {B^T G B} vs full {B0^T G B1}; is shared(MANIFOLD).project(v) tied?"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.shared_geometry as sg
np.random.seed(3)
M = t3m.MANIFOLD
N, n = 5, 2
x = t3.TuckerTensorTrain.randn((N, N), (n, n), (1, n, 1)).share((0, 0)); S = sg.shared_manifold((0, 0)); f = S.frame(x)
B = np.asarray(x.tucker_cores[0]); G = np.einsum('ear,rbf->ab', x.tt_cores[0], x.tt_cores[1])
assert np.allclose(B.T @ G @ B, x.to_dense())
E = lambda shp, idx: (lambda Z: (Z.__setitem__(idx, 1.0), Z)[1])(np.zeros(shp))
J_tied = np.stack([(E(B.shape, i).T @ G @ B + B.T @ G @ E(B.shape, i)).ravel() for i in np.ndindex(B.shape)]
                  + [(B.T @ E(G.shape, i) @ B).ravel() for i in np.ndindex(G.shape)], axis=1)
J_full = np.stack([(E(B.shape, i).T @ G @ B).ravel() for i in np.ndindex(B.shape)] + [(B.T @ G @ E(B.shape, i)).ravel() for i in np.ndindex(B.shape)]
                  + [(B.T @ E(G.shape, i) @ B).ravel() for i in np.ndindex(G.shape)], axis=1)
sv_t, sv_f = np.linalg.svd(J_tied, compute_uv=False), np.linalg.svd(J_full, compute_uv=False)
print('tied-set tangent dim =', int((sv_t > 1e-10 * sv_t[0]).sum()), ' full-manifold tangent dim =', int((sv_f > 1e-10 * sv_f[0]).sum()),
      '| library manifold_dim: full', t3m.manifold_dim(((N, N), (n, n), (1, n, 1))), 'shared', t3m.manifold_dim(((N, N), (n, n), (1, n, 1)), sharing=(0, 0)))
def dist(vec, J):
    c, *_ = np.linalg.lstsq(J, vec, rcond=None); return np.linalg.norm(J @ c - vec) / np.linalg.norm(vec)
for k in range(3):
    v = M.randn(f); pv = S.project(v)
    print('sample %d: raw gauged tangent dist to TIED space=%.2e (to FULL=%.2e) | S.project(v) dist to TIED=%.2e | |pv-v|/|v|=%.2e'
          % (k, dist(v.to_dense().ravel(), J_tied), dist(v.to_dense().ravel(), J_full), dist(pv.to_dense().ravel(), J_tied), np.linalg.norm(pv.to_dense() - v.to_dense()) / np.linalg.norm(v.to_dense())))
g = np.random.randn(N, N); pa = S.project_ambient(f, g); pm = M.project_ambient(f, g)
print('S.project_ambient(g): dist to TIED=%.2e, to FULL=%.2e | MANIFOLD.project_ambient(g): dist to TIED=%.2e' % (dist(pa.to_dense().ravel(), J_tied), dist(pa.to_dense().ravel(), J_full), dist(pm.to_dense().ravel(), J_tied)))
# and does the exact tied projection of g differ from the full one?
ct, *_ = np.linalg.lstsq(J_tied, g.ravel(), rcond=None); cf, *_ = np.linalg.lstsq(J_full, g.ravel(), rcond=None)
print('|Pi_tied g - Pi_full g| / |Pi_full g| (from Jacobians) = %.2e ; |S.project_ambient(g) - MANIFOLD.project_ambient(g)| / |.| = %.2e'
      % (np.linalg.norm(J_tied @ ct - J_full @ cf) / np.linalg.norm(J_full @ cf), np.linalg.norm(pa.to_dense() - pm.to_dense()) / np.linalg.norm(pm.to_dense())))
