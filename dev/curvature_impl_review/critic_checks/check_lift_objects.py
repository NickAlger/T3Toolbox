"""Read-only check: the lift's S_i, T_i vs objects the library already computes.

T_i := einsum('abi,abj->ij', L_i, H_i^X) should equal the right-to-left zipper Z_{i+1} of the left
chain against the right chain (for i <= d-2, L_i left-orthogonal); S_i^T = <O_i, H_i> (sharing.py);
and the construction's tucker_variations V_i^X = S_i^T U_i.
"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto

np.random.seed(0)
for struct in [((5, 6, 7, 5), (2, 3, 3, 2), (1, 2, 3, 2, 1)),
               ((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))]:       # second one non-minimal (slack)
    x = t3.TuckerTensorTrain.randn(*struct)
    frame, var = fvc.t3_orthogonal_representations(x.data)
    U, O, L, R = frame
    V, H = var
    d = len(U)
    ZZ = tto.tt_zipper_right_to_left(L, R)
    HH = [np.einsum('aib,bc->aic', L[i], ZZ[i + 1]) for i in range(d)]
    print('struct', struct)
    print('  up ranks', [u.shape[-2] for u in U], 'down ranks', [o.shape[-2] for o in O])
    print('  left', [l.shape[-1] for l in L], 'right', [r.shape[-3] for r in R])
    print('  centers H (construction) == L_i Z_{i+1}:', all(np.allclose(a, b) for a, b in zip(H, HH)))
    for i in range(d - 1):
        T = np.einsum('abi,abj->ij', L[i], H[i])
        print('  T_%d shape %s == Z_%d: %s  svals %s' % (i, T.shape, i + 1, np.allclose(T, ZZ[i + 1]),
              np.round(np.linalg.svd(T, compute_uv=False), 4)))
    for i in range(d):
        ST = np.einsum('axb,aub->xu', O[i], H[i])
        print('  S_%d^T shape %s ; V_i == S_i^T U_i: %s' % (i, ST.shape, np.allclose(V[i], ST @ U[i])))
