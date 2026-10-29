"""Read-only plan check (v5, I-18a fix): a CANONICAL zero frame. (1) Is the ragged frame at the zero tensor the
identity-prefix completion (up/left/right = [I;0]-type orthonormal cores, down = 0)? (2) Pack THAT ragged frame into
the uniform layout (t3frame_to_ut3frame, no SVD) and evaluate the uniform kind's transpose + the uniform geometry's
project/inner there: does the uniform reference equal the ragged one (layer-independent reference)? (3) Shared."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf, t3toolbox.backend.ufv_conversions as ufc
import t3toolbox.corewise as cw
rng = np.random.default_rng(0); np.random.seed(0)
def is_prefix_identity(M):          # M: (rows, cols) -- columns orthonormal and equal to an identity prefix
    k = min(M.shape); E = np.zeros(M.shape); E[np.arange(k), np.arange(k)] = 1.0
    return np.allclose(M, E)
for shape, tk, tt in (((5, 6, 7), (2, 3, 2), (1, 2, 2, 1)), ((6, 6, 6, 6), (3, 3, 3, 3), (1, 3, 4, 3, 1))):
    W = 40
    ww = [rng.standard_normal((W, N)) for N in shape]
    data = np.asarray(t3.TuckerTensorTrain.randn(shape, tk, tt).apply(ww))
    zero = t3.TuckerTensorTrain.zeros(shape, tk, tt)
    geom = bgeo.ManifoldGeometryOps()
    prob = bopt.least_squares_problem(geom, bfit.APPLY, ww, data)
    lm = prob.local_model(zero.data); U, O, L, R = lm.frame
    upre = all(is_prefix_identity(u.T) for u in U)
    lpre = all(is_prefix_identity(l.reshape(-1, l.shape[-1])) for l in L)
    rpre = all(is_prefix_identity(r.reshape(r.shape[0], -1).T) for r in R)
    ozero = all(np.all(o == 0) for o in O)
    g = lm.gradient; ref_r = float(geom.inner(g, g)) ** 0.5
    print(f"{shape}: ragged zero frame identity-prefix? up {upre} left {lpre} right {rpre}; down == 0 {ozero}; ragged ref {ref_r:.6e}")
    # pack the ragged canonical frame (no SVD) and evaluate on the uniform layer
    uz = ut3.UniformTuckerTensorTrain.from_t3(zero)
    ufr = ufc.t3frame_to_ut3frame(lm.frame, N=uz.data[0].shape[-1])
    ug = bgeo.UniformManifoldGeometryOps.from_point(uz.data)
    kind = uf.uniform_sampling_kind('apply', uz.data, None)
    psample = uf.pack_sample('apply', ww, uz.N)
    sweep = kind.precompute(ufr, psample)
    resid = -data
    gu = ug.project(ufr, kind.transpose(resid, psample, ufr, sweep))
    try:
        ref_u = float(ug.inner(gu, gu)) ** 0.5
        print(f"   uniform at the PACKED canonical frame: ref {ref_u:.6e} (rel diff {abs(ref_u - ref_r) / ref_r:.1e});"
              f" uniform at its own SVD frame: {float(ug.inner(*(2 * (bopt.least_squares_problem(ug, kind, psample, data).local_model((uz.data[0], uz.data[1])).gradient,)))) ** 0.5:.6e}")
    except Exception as e:
        print('   uniform packed-frame evaluation failed:', type(e).__name__, str(e)[:200])
