"""R4-3: tv_oblique_gauge_projection -- is it a projection (idempotent), does it preserve the represented
vector, land in the gauged subspace, and annihilate pure-gauge (vertical) variations?"""
import itertools
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.backend.tv_operations as tv
from r4_common import STRUCTS, NONMIN, relerr

np.random.seed(1)
worst = {}
def note(k, e): worst[k] = max(worst.get(k, 0.0), e)

for (shape, tr, ttr), K, C in itertools.product(STRUCTS + [NONMIN], [(), (2,)], [(), (3,)]):
    d = len(shape)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
    frame = t3m.MANIFOLD.frame(x)
    v = t3m.COREWISE.randn(frame, stack_shape=K)
    key = f'd={d} tr={tr} K={K} C={C}'
    fd, vd = frame.data, v.variations.data
    ob = tv.tv_oblique_gauge_projection(fd, vd)
    # (a) preserves the represented vector
    note(key + ' preserves dense', relerr(tv.tv_to_dense(fd, vd), tv.tv_to_dense(fd, ob)))
    # (b) result gauged
    note(key + ' gauged', float(np.max(tv.tv_gauge_residual(fd, ob))))
    # (c) idempotent
    note(key + ' idempotent', cw.corewise_relerr(ob, tv.tv_oblique_gauge_projection(fd, ob)))
    # (d) annihilates a pure-gauge variation (represents the zero tangent): V_i = X U_i, H_i -= O_i X  and
    #     H_i += L_i Y, H_{i+1} -= Y R_{i+1}
    U, O, L, R = fd
    VV = [np.zeros_like(c) for c in vd[0]]; HH = [np.zeros_like(c) for c in vd[1]]
    for i in range(d):
        X = np.random.randn(*(K + C + (O[i].shape[-2], U[i].shape[-2])))       # (nD, nU)
        VV[i] = VV[i] + np.einsum('...ji,...io->...jo', X, U[i])
        HH[i] = HH[i] - np.einsum('...aib,...ij->...ajb', O[i], X)
    for i in range(d - 1):
        Y = np.random.randn(*(K + C + (L[i].shape[-1], R[i + 1].shape[-3])))   # (rL(i+1), rR(i+1))
        HH[i] = HH[i] + np.einsum('...iaj,...jk->...iak', L[i], Y)
        HH[i + 1] = HH[i + 1] - np.einsum('...jk,...kbl->...jbl', Y, R[i + 1])
    gd = (tuple(VV), tuple(HH))
    zero_dense = tv.tv_to_dense(fd, gd)
    note(key + ' vertical represents 0 (sanity)', float(np.max(np.abs(zero_dense))) / max(1.0, float(cw.corewise_norm(gd))))
    note(key + ' kills vertical', float(cw.corewise_norm(tv.tv_oblique_gauge_projection(fd, gd))) / float(cw.corewise_norm(gd)))
    # (e) v - Pv is vertical (represents zero): already implied by (a); and  P(orth-gauge(v)) == orth-gauge(v)
    og = tv.tv_orthogonal_gauge_projection(fd, vd)
    note(key + ' fixes gauged input', cw.corewise_relerr(og, tv.tv_oblique_gauge_projection(fd, og)))

# on a NON-orthogonal (corewise) frame: the vector is still preserved but the gauge is not attained
x = t3.TuckerTensorTrain.randn(*STRUCTS[2])
cf = t3m.COREWISE.frame(x); v = t3m.COREWISE.randn(cf)
ob = tv.tv_oblique_gauge_projection(cf.data, v.variations.data)
print('corewise frame: preserves dense =', relerr(tv.tv_to_dense(cf.data, v.variations.data), tv.tv_to_dense(cf.data, ob)),
      ' gauge residual =', float(np.max(tv.tv_gauge_residual(cf.data, ob))))

bad = {k: e for k, e in worst.items() if not (e < 1e-9)}
print('checked', len(worst), 'quantities; max =', max(worst.values()))
print('FAILURES:' if bad else 'oblique projection: all properties hold', bad)
