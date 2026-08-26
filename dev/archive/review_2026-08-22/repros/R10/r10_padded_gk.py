"""R10: the GK metric route on a train padded ABOVE its real rank (the rank-continuation situation)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubvf
import t3toolbox.uniform_manifold as ut3m
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
for n, r in ((3, 3), (4, 4)):
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=n, r=r)
    frame, _ = ubvf.ut3_orthogonal_representations(ux)
    W = ut3.UT3Weights.from_ut3svd(ux)
    gk = ubvf.UT3FrameWeights.from_ut3weights(W).reciprocal()
    v = ut3m.UNIFORM_COREWISE.randn(frame)
    print('ux padded (n,r)=(%d,%d): frame (nU,nD,rL,rR)=%s ; from_ut3svd W (n,r)=(%d,%d) consistent_with(ux)=%s ; gk widths=%s' % (
        n, r, (frame.nU, frame.nD, frame.rL, frame.rR), W.n, W.r, W.is_consistent_with(ux), (gk.nU, gk.nD, gk.rL, gk.rR)))
    for label, f in (('ut3_weighted_norm(ux, W)', lambda: ut3.ut3_weighted_norm(ux, W)), ('tangent.weighted_norm(gk)', lambda: v.weighted_norm(gk))):
        try: f(); print('   %-28s OK' % label)
        except Exception as e: print('   %-28s RAISED %s: %s' % (label, type(e).__name__, str(e).splitlines()[-1][:100]))
    # the ragged detour that does work
    W2 = ut3.UT3Weights.from_t3weights(t3.T3Weights.from_t3svd(ux.to_t3()), n=ux.n, r=ux.r)
    print('   ragged detour from_t3weights(from_t3svd(ux.to_t3()), n=ux.n, r=ux.r) consistent:', W2.is_consistent_with(ux), end='; ')
    try: v.weighted_norm(ubvf.UT3FrameWeights.from_ut3weights(W2).reciprocal()); print('tangent.weighted_norm OK')
    except Exception as e: print('tangent.weighted_norm RAISED', type(e).__name__)
