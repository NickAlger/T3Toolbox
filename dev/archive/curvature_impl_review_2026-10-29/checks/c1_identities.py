import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv

np.random.seed(0)
for struct in [((5,6,7,5),(2,3,3,2),(1,2,3,2,1)), ((6,6,6),(4,4,4),(1,2,2,1))]:
    x = t3.TuckerTensorTrain.randn(*struct)
    frame, var = fvc.t3_orthogonal_representations(x.data)
    U, O, L, R = frame
    V, H = var
    Z = tto.tt_zipper_right_to_left(L, R)
    d = len(U)
    print('struct', struct, 'up', [u.shape for u in U], 'down', [o.shape for o in O])
    print(' left', [l.shape for l in L], 'right', [r.shape for r in R])
    for i in range(d):
        Hi = np.einsum('aib,bc->aic', L[i], Z[i+1])
        ST = np.einsum('axb,aub->xu', O[i], H[i])
        STb = V[i] @ U[i].T
        if i < d - 1:     # T_i is a lift factor only for i <= d-2 (the last TT slot lifts undamped, no T)
            Ti = np.einsum('abi,abj->ij', L[i], H[i])
            tz = '|T-Z|=%.1e T.shape=%s' % (np.linalg.norm(Ti - Z[i+1]), Ti.shape)
        else:
            tz = '(no T_{d-1}: last slot)'
        print('  i=%d |H-LZ|=%.1e %s |ST-VU^T|=%.1e ST.shape=%s' % (
            i, np.linalg.norm(Hi-H[i]), tz, np.linalg.norm(ST-STb), ST.shape))
    # singular values vs dense unfoldings
    X = x.to_dense()
    for i in range(d):
        Xi = np.moveaxis(X, i, 0).reshape(X.shape[i], -1)
        s_dense = np.linalg.svd(Xi, compute_uv=False)
        ST = np.einsum('axb,aub->xu', O[i], H[i])
        s = np.linalg.svd(ST, compute_uv=False)
        k = len(s)
        print('  tucker sv mode %d match: %s' % (i, np.allclose(s, s_dense[:k])))
    for i in range(d-1):
        Xi = X.reshape(int(np.prod(X.shape[:i+1])), -1)
        s_dense = np.linalg.svd(Xi, compute_uv=False)
        s = np.linalg.svd(Z[i+1], compute_uv=False)
        print('  bond sv %d match: %s (Zshape %s)' % (i+1, np.allclose(s, s_dense[:len(s)]), Z[i+1].shape))
    print('  Z[d]', Z[d])
