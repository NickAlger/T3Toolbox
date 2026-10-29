"""Read-only plan check: the pinv lift on a TT-non-minimal frame (left/right bond ranks differ): where does
DPhi[L+ v] = E v fail?  Expect: exact for the Tucker slots and tall T, lossy on a WIDE T_i (rL < rR)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
rng = np.random.default_rng(3); np.random.seed(3)
x = t3.TuckerTensorTrain.randn((6, 6, 6), (4, 4, 4), (1, 9, 9, 1))
frame, var0 = fvc.t3_orthogonal_representations(x.data)
U, O, L, R = frame
d = 3
print('left ranks', [l.shape[-1] for l in L], 'right ranks', [r.shape[-3] for r in R])
Z = tto.tt_zipper_right_to_left(L, R)
cf = fvc.t3_corewise_frame((U, L))
for i in range(d - 1):
    vg = [np.zeros(h.shape) for h in var0[1]]; vg[i] = rng.standard_normal(var0[1][i].shape)
    vt = [np.zeros(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)]
    v = tv.tv_orthogonal_gauge_projection(frame, (tuple(vt), tuple(vg)))
    th = (tuple(np.zeros_like(u) for u in U), tuple(np.einsum('aib,bc->aic', v[1][k], np.linalg.pinv(Z[k + 1])) if k < d - 1 else v[1][k] for k in range(d)))
    th = (tuple(np.linalg.pinv(np.einsum('axb,aub->xu', O[k], np.einsum('aib,bc->aic', L[k], Z[k + 1]))) @ v[0][k] for k in range(d)), th[1])
    a, b = tv.tv_to_dense(frame, v), tv.tv_to_dense(cf, th)
    print('TT slot %d: H shape %s T shape %s  lift identity rel err %.1e' % (i, var0[1][i].shape, Z[i + 1].shape, np.linalg.norm(a - b) / np.linalg.norm(a)))
