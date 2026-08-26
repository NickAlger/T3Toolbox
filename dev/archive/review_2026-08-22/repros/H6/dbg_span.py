import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.manifold as t3m, t3toolbox.shared_geometry as sg
np.random.seed(3); M = t3m.MANIFOLD; N, n = 5, 2
x = t3.TuckerTensorTrain.randn((N, N), (n, n), (1, n, 1)).share((0, 0)); S = sg.shared_manifold((0, 0)); f = S.frame(x)
B = np.asarray(x.tucker_cores[0]); G = np.einsum('ear,rbf->ab', x.tt_cores[0], x.tt_cores[1]); print('B', B.shape, 'G', G.shape, 'tucker_cores[0] is [1]:', x.tucker_cores[0] is x.tucker_cores[1])
def E(shp, idx): Z = np.zeros(shp); Z[idx] = 1.0; return Z
Jt = np.stack([(E(B.shape, i).T @ G @ B + B.T @ G @ E(B.shape, i)).ravel() for i in np.ndindex(B.shape)] + [(B.T @ E(G.shape, i) @ B).ravel() for i in np.ndindex(G.shape)], axis=1)
Jf = np.stack([(E(B.shape, i).T @ G @ B).ravel() for i in np.ndindex(B.shape)] + [(B.T @ G @ E(B.shape, i)).ravel() for i in np.ndindex(B.shape)] + [(B.T @ E(G.shape, i) @ B).ravel() for i in np.ndindex(G.shape)], axis=1)
print('Jt', Jt.shape, 'rank', np.linalg.matrix_rank(Jt), '| Jf', Jf.shape, 'rank', np.linalg.matrix_rank(Jf), '| manifold_dim full', t3m.manifold_dim(((N, N), (n, n), (1, n, 1))), 'shared', t3m.manifold_dim(((N, N), (n, n), (1, n, 1)), sharing=(0, 0)))
def dist(vec, J):
    c, *_ = np.linalg.lstsq(J, vec, rcond=None); return np.linalg.norm(J @ c - vec) / np.linalg.norm(vec)
for k in range(3):
    v = M.randn(f); pv = S.project(v)
    print('raw v: dist TIED %.2e FULL %.2e | S.project(v): dist TIED %.2e | |pv-v|/|v| %.2e' % (dist(v.to_dense().ravel(), Jt), dist(v.to_dense().ravel(), Jf), dist(pv.to_dense().ravel(), Jt), np.linalg.norm(pv.to_dense()-v.to_dense())/np.linalg.norm(v.to_dense())))
g = np.random.randn(N, N); pa = S.project_ambient(f, g); pm = M.project_ambient(f, g)
ct, *_ = np.linalg.lstsq(Jt, g.ravel(), rcond=None); cf, *_ = np.linalg.lstsq(Jf, g.ravel(), rcond=None)
print('exact Pi_tied g vs Pi_full g rel diff %.2e | S.project_ambient vs MANIFOLD.project_ambient rel diff %.2e | S.project_ambient dist TIED %.2e | exact Pi_tied g vs S.project_ambient %.2e'
      % (np.linalg.norm(Jt@ct - Jf@cf)/np.linalg.norm(Jf@cf), np.linalg.norm(pa.to_dense()-pm.to_dense())/np.linalg.norm(pm.to_dense()), dist(pa.to_dense().ravel(), Jt), np.linalg.norm(Jt@ct - pa.to_dense().ravel())/np.linalg.norm(Jt@ct)))
