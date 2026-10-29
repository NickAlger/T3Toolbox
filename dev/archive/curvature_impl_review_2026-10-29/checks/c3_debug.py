import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.t3_constructors as t3k
rng = np.random.default_rng(1)
shape, tr, ttr = (5, 6, 6, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)
x = t3k.t3_corewise_randn(shape, tr, ttr)
print(type(x), len(x), [type(a) for a in x])
frame, var0 = fvc.t3_orthogonal_representations(x)
U, O, L, R = frame
print([u.shape for u in U], [o.shape for o in O])
d = 4
cframe = fvc.t3_corewise_frame((tuple(U), tuple(L)))
vshapes = ([o.shape[1:2] + u.shape[1:] for o, u in zip(O, U)], [h.shape for h in L])
v_raw = (tuple(rng.standard_normal(s) for s in vshapes[0]), tuple(rng.standard_normal(s) for s in vshapes[1]))
print('norm dense frame raw', np.linalg.norm(tv.tv_to_dense(frame, v_raw)))
# single-slot tests
Z = tto.tt_zipper_right_to_left(L, R)
for i in range(d):
    for kind in ('tucker', 'tt'):
        vt = [np.zeros_like(a) for a in v_raw[0]]; vg = [np.zeros_like(a) for a in v_raw[1]]
        tht = [np.zeros_like(u) for u in U]; thg = [np.zeros_like(l) for l in L]
        if kind == 'tucker':
            vt[i] = v_raw[0][i]
            ST = np.einsum('axb,aub->xu', O[i], var0[1][i])
            tht[i] = np.linalg.pinv(ST) @ vt[i]
        else:
            vg[i] = v_raw[1][i]
            thg[i] = np.einsum('aib,bc->aic', vg[i], np.linalg.pinv(Z[i + 1])) if i < d - 1 else vg[i]
        a = tv.tv_to_dense(frame, (tuple(vt), tuple(vg)))
        b = tv.tv_to_dense(cframe, (tuple(tht), tuple(thg)))
        print(i, kind, 'rel', np.linalg.norm(a - b) / np.linalg.norm(a), np.linalg.norm(a))
