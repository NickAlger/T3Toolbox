"""docs/transposes.md: 'corewise = the ambient back-projection PROJECTED onto the core-parametrized tangent
space at X (the span of single-core perturbations)'. Check: is J g_corewise (the corewise gradient pushed to a
tensor) equal to the orthogonal projection of the ambient tensor A onto range(J)?  g_corewise = J^T A."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
np.random.seed(0)
shape, tr, rr = (4, 3), (2, 2), (1, 2, 1)
x = t3.TuckerTensorTrain.randn(shape, tr, rr)
tk, tt = [list(c) for c in x.data]
ww = [np.random.randn(N) for N in shape]
# corewise Jacobian J as a dense matrix (columns = vec(single-core perturbation tensor))
cols = []
def replace(kind, i, new):
    a, b = list(tk), list(tt); (a if kind == 'U' else b)[i] = new
    return t3.TuckerTensorTrain(tuple(a), tuple(b)).to_dense().ravel()
for kind, cores in (('U', tk), ('G', tt)):
    for i, c in enumerate(cores):
        for k in range(c.size):
            e = np.zeros(c.size); e[k] = 1.0
            cols.append(replace(kind, i, e.reshape(c.shape)))
J = np.stack(cols, axis=1)                                   # (prod N, n_params)
A = t3.TuckerTensorTrain.from_canonical(t3.TuckerTensorTrain.apply_ambient_transpose(np.asarray(1.3), ww)).to_dense().ravel()
gU, gG = x.apply_corewise_transpose(np.asarray(1.3), ww, sum_over_probes=True)
g = np.concatenate([a.ravel() for a in gU + gG])
print('corewise gradient == J^T A (pullback):        ', np.allclose(g, J.T @ A))
P_A = J @ np.linalg.pinv(J) @ A                              # orthogonal projection of A onto range(J)
print('J g == orthogonal projection of A onto range(J):', np.allclose(J @ g, P_A), '  relerr =', np.linalg.norm(J @ g - P_A) / np.linalg.norm(P_A))
print('(J^T J == I would make them agree; ||J^T J - I|| =', np.linalg.norm(J.T @ J - np.eye(J.shape[1])), ')')
