import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.t3_conversions as t3c
rng = np.random.default_rng(11)
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))
frame, var0 = fvc.t3_orthogonal_representations(x.data)
U, O, L, R = frame
d = 3
print('Phi(U,L) == X:', np.allclose(t3c.t3_to_dense((U, L)), x.to_dense()))
Z = tto.tt_zipper_right_to_left(L, R)
cf = fvc.t3_corewise_frame((U, L))
for i in range(d):
    Hc = np.einsum('aib,bc->aic', L[i], Z[i + 1])
    print(i, 'H==LZ', np.allclose(Hc, var0[1][i]), 'S^T', np.einsum('axb,aub->xu', O[i], Hc).shape,
          'rank', np.linalg.matrix_rank(np.einsum('axb,aub->xu', O[i], Hc)), 'H=O S^T:',
          np.allclose(np.einsum('axb,xu->aub', O[i], np.einsum('axb,aub->xu', O[i], Hc)), Hc))
    for kind in ('tucker', 'tt'):
        vt = [np.zeros(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)]; vg = [np.zeros(l.shape) for l in L]
        tht = [np.zeros_like(u) for u in U]; thg = [np.zeros_like(l) for l in L]
        if kind == 'tucker':
            vt[i] = rng.standard_normal(vt[i].shape)
            tht[i] = np.linalg.pinv(np.einsum('axb,aub->xu', O[i], Hc)) @ vt[i]
        else:
            vg[i] = rng.standard_normal(vg[i].shape)
            thg[i] = np.einsum('aib,bc->aic', vg[i], np.linalg.pinv(Z[i + 1])) if i < d - 1 else vg[i]
        a = tv.tv_to_dense(frame, (tuple(vt), tuple(vg)))
        b = tv.tv_to_dense(cf, (tuple(tht), tuple(thg)))
        print('   ', kind, 'rel', np.linalg.norm(a - b) / np.linalg.norm(a))
