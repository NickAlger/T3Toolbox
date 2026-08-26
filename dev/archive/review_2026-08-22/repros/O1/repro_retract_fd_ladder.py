"""Per-element FD ladder for MANIFOLD.retract at the C=(2,3) case that failed the h^2 check in the sweep."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
for sname, (shape, tr, ttr) in {'d2': ((3, 5), (2, 3), (1, 2, 1)), 'd3': ((3, 5, 4), (2, 3, 2), (1, 2, 3, 1))}.items():
    C = (2, 3)
    np.random.seed(1)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
    frame, _ = bvf.t3_orthogonal_representations(x)
    np.random.seed(4)
    v = t3m.MANIFOLD.randn(frame)
    Vd = np.asarray(v.to_dense())
    _, stk, stt = x.t3svd()
    print(sname, 'min tucker sval per element:', np.round(np.array([np.min(np.asarray(s), axis=-1) for s in stk]).min(axis=0), 3).reshape(-1))
    print(sname, 'min TT sval per element:    ', np.round(np.array([np.min(np.asarray(s), axis=-1) for s in stt[1:-1]]).min(axis=0), 3).reshape(-1))
    for h in (1e-2, 5e-3, 1e-3, 5e-4, 1e-4):
        rp = np.asarray(t3m.MANIFOLD.retract(v * h).to_dense()); rm = np.asarray(t3m.MANIFOLD.retract(v * (-h)).to_dense())
        fd = (rp - rm) / (2 * h)
        per = np.linalg.norm((fd - Vd).reshape(6, -1), axis=-1) / np.linalg.norm(Vd.reshape(6, -1), axis=-1)
        print('  h=%.0e per-element relerr: %s' % (h, np.array2string(per, precision=2)))
