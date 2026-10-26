"""Read-only plan check (v4, m29 + m30).
(a) m29: the uniform kind can reach the frame's (um, lm) through the FRAME argument (frame_data[5]) -- the uniform
    sweep's mb holds bare supercores only; masking theta = (frame[0], frame[2]) with (um, lm) via ut3_apply_masks
    reproduces the sweep's masked (mb[0], mb[2]) exactly (so theta and the sweep agree), on manifold AND corewise
    uniform frames, incl. TT slack.
(b) m30: on the numpy path float32 inputs are upcast (t3_apply -> float64 residual), so eps must come from the
    RESIDUAL's dtype (float64 there; float32 only on the jax-without-x64 path)."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.geometry as bgeo, t3toolbox.backend.ufv_masking as fm, t3toolbox.backend.ut3_masking as um_
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.fitting as bfit, t3toolbox.backend.t3_constructors as t3k
np.random.seed(2)
for label, x in (('minimal', t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))),
                 ('TT slack', t3.TuckerTensorTrain.randn((6, 6, 6), (4, 4, 4), (1, 9, 9, 1)))):
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=5, r=10, N=8)
    for gname, G in (('manifold', bgeo.UniformManifoldGeometryOps), ('corewise', bgeo.UniformCorewiseGeometryOps)):
        g = G.from_point(ux.data)
        fr = g.frame((ux.data[0], ux.data[1]))
        mb = fm.ufv_apply_frame_masks(fr)
        um, lm = fr[5][0], fr[5][2]
        mtk, mtt = um_.ut3_apply_masks((fr[0], fr[2], fr[4], (um, lm)))
        print('%-9s %-8s sweep mb holds masks: %s ; ut3_apply_masks(theta, frame (um,lm)) == (mb[0], mb[2]): %s' % (
            label, gname, any(a.dtype == bool for a in mb), np.array_equal(mtk, mb[0]) and np.array_equal(mtt, mb[2])))
x = t3k.t3_corewise_randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
x32 = (tuple(a.astype(np.float32) for a in x[0]), tuple(a.astype(np.float32) for a in x[1]))
rng = np.random.default_rng(0)
ww = [rng.standard_normal((20, N)).astype(np.float32) for N in (5, 6, 7)]
prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, rng.standard_normal(20).astype(np.float32))
print('numpy, float32 inputs: residual dtype', np.asarray(prob.local_model(x32).residual).dtype, '-> eps', np.finfo(np.asarray(prob.local_model(x32).residual).dtype).eps)
