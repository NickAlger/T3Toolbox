"""UniformTuckerTensorTrain.norm() / UNIFORM_MANIFOLD.norm with forced (extra) padding."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(1)
for C in [(), (2,), (2, 3)]:
    x = t3.TuckerTensorTrain.randn((3, 5), (2, 3), (1, 2, 1), stack_shape=C)
    for pad in [{}, dict(N=8, n=5, r=5)]:
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, **pad)
        ref = np.asarray(x.norm())
        for label, fn in [('ux.norm()', lambda: ux.norm()), ('ux.norm(use_orthogonalization=False)', lambda: ux.norm(use_orthogonalization=False)),
                          ('sqrt(ux.inner(ux))', lambda: np.sqrt(ux.inner(ux))), ('UNIFORM_MANIFOLD.norm(randn)', None)]:
            try:
                if fn is None:
                    frame, _ = ubv.ut3_orthogonal_representations(ux)
                    v = ut3m.UNIFORM_MANIFOLD.randn(frame)
                    got = np.asarray(ut3m.UNIFORM_MANIFOLD.norm(v)); ref2 = np.linalg.norm(np.asarray(v.to_dense()).reshape(C + (-1,)), axis=-1)
                    print('C=%-6s pad=%-22s %-38s relerr=%.2e' % (C, pad, label, np.linalg.norm(got - ref2) / np.linalg.norm(ref2)))
                else:
                    got = np.asarray(fn())
                    print('C=%-6s pad=%-22s %-38s relerr=%.2e  (got %s, ref %s)' % (C, pad, label, np.linalg.norm(got - ref) / np.linalg.norm(ref), np.round(got, 3).tolist(), np.round(ref, 3).tolist()))
            except Exception as e:
                print('C=%-6s pad=%-22s %-38s RAISES %s: %s' % (C, pad, label, type(e).__name__, e))
