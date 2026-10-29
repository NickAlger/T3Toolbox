"""Read-only plan check (v5, S0c): (1) E7 reproduced -- the uniform COREWISE kind transpose leaves garbage in padded
output slots; (2) re-masking the output with the frame's gauge-shifted variation masks zeroes it and leaves the real block
bit-identical; (3) its cost relative to the transpose itself (numpy, eager)."""
import sys, time; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.geometry as bgeo, t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.ufv_masking as ufm
def tm(f, n=5):
    f(); t0 = time.perf_counter()
    for _ in range(n):
        out = f()
    return (time.perf_counter() - t0) / n, out
rng = np.random.default_rng(0); np.random.seed(0)
for shape, tk, tt, W, order in (((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), 80, 2), ((12, 14, 16, 10), (4, 6, 5, 3), (1, 4, 6, 3, 1), 400, 2)):
    x = t3.TuckerTensorTrain.randn(shape, tk, tt); ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    ww = [rng.standard_normal((W, N)) for N in shape]; pp = [rng.standard_normal((W, N)) for N in shape]
    tgt = t3.TuckerTensorTrain.randn(shape, tk, tt)
    kind_r = uf.uniform_derivatives_kind('probe_derivatives', ux.data, order, None)
    import t3toolbox.backend.fitting as bfit
    data = [np.asarray(z) for z in bfit.probe_derivatives_kind(order, None).point_forward(tgt.data, (ww, pp))]
    for gname, cls in (('UNIFORM_COREWISE', bgeo.UniformCorewiseGeometryOps), ('UNIFORM_MANIFOLD', bgeo.UniformManifoldGeometryOps)):
        geom = cls.from_point(ux.data)
        prob = bopt.least_squares_problem(geom, kind_r, uf.pack_sample('probe_derivatives', (ww, pp), ux.N),
                                          uf.pack_data('probe_derivatives', data, ux.N))
        lm = prob.local_model((ux.data[0], ux.data[1]))
        t_T, raw = tm(lambda: lm.kind.transpose(lm.residual, lm.sample, lm.frame, lm.sweep))
        vm = ufm.ufv_variation_masks(lm.frame[5])
        vdata = (raw[0], raw[1], lm.frame[4], vm)
        t_M, masked = tm(lambda: ufm.ufv_apply_variations_masks(vdata))
        pad = max(float(np.abs(a - b).max()) for a, b in zip(raw, masked))
        real_same = all(np.array_equal(np.where(b != 0, a, 0), b) for a, b in zip(raw, masked))
        print(f"{shape} {gname}: max |padded output| before re-mask {pad:.1e}; real block bit-identical {real_same}; "
              f"re-mask {1e3 * t_M:.2f} ms vs transpose {1e3 * t_T:.2f} ms ({100 * t_M / t_T:.2f}%)")
